import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression
from lifelines.utils import concordance_index
import warnings

warnings.filterwarnings('ignore')

print("Executing V77 'THE SPECIALIST STACK' (Professional Stacking Ensemble)...")

# --- 1. Data and Features ---
train_raw = pd.read_csv("dataset/train.csv").fillna(0)
test_raw = pd.read_csv("dataset/test.csv").fillna(0)

def engineer_features(df):
    df = df.copy()
    df['growth_momentum'] = df['area_growth_rate_ha_per_h'] * df['dt_first_last_0_5h']
    df['threat_alignment'] = df['closing_speed_m_per_h'] * df['alignment_abs']
    df['is_extreme_close'] = (df['dist_min_ci_0_5h'] < 5000).astype(int)
    df['log_dist'] = np.log1p(df['dist_min_ci_0_5h'])
    return df

train = engineer_features(train_raw)
test = engineer_features(test_raw)

base_features = [
    'dist_min_ci_0_5h', 'log_dist', 'num_perimeters_0_5h', 
    'alignment_abs', 'threat_alignment', 'growth_momentum', 
    'is_extreme_close', 'area_growth_rate_ha_per_h', 'closing_speed_m_per_h',
    'dt_first_last_0_5h', 'area_first_ha', 'centroid_speed_m_per_h'
]

# --- 2. Step 1: The Meta-Feature (Specialist Arrival Ranking) ---
print("Generating Specialist Meta-Features...")
hits_idx = train[train['event'] == 1].index
spec_feats = ['closing_speed_m_per_h', 'alignment_abs', 'area_growth_rate_ha_per_h', 'centroid_speed_m_per_h', 'dist_min_ci_0_5h', 'along_track_speed']

# Cross-validated meta-features for training set
train['spec_score'] = 0.0
kf_spec = KFold(n_splits=5, shuffle=True, random_state=42)
for tr_idx, val_idx in kf_spec.split(train):
    # Only train on hitters within the fold
    fold_hits = [i for i in tr_idx if i in hits_idx]
    if len(fold_hits) > 0:
        m_spec = xgb.XGBRegressor(n_estimators=50, max_depth=2, random_state=42)
        m_spec.fit(train.iloc[fold_hits][spec_feats], train.iloc[fold_hits]['time_to_hit_hours'])
        # Predict for ALL rows in the validation fold
        train.loc[val_idx, 'spec_score'] = m_spec.predict(train.iloc[val_idx][spec_feats])

# Meta-features for test set
m_spec_final = xgb.XGBRegressor(n_estimators=50, max_depth=2, random_state=42)
m_spec_final.fit(train.iloc[hits_idx][spec_feats], train.iloc[hits_idx]['time_to_hit_hours'])
test['spec_score'] = m_spec_final.predict(test[spec_feats])

# --- 3. Step 2: The Master Model (50-Seed Bagging) ---
print("Training Master Model (12 Features + 1 Meta-Feature, 50 Seeds)...")
features = base_features + ['spec_score']
y_time, y_event = train['time_to_hit_hours'], train['event']
y_xgb = y_time * np.where(y_event == 1, 1, -1)

def get_breslow(m, X, X_tr, yt_tr, ye_tr):
    ut = np.sort(np.unique(yt_tr)); h0 = np.zeros_like(ut); m_tr = m.predict(X_tr)
    for i, t in enumerate(ut):
        rs = yt_tr >= t
        h0[i] = np.sum((yt_tr == t) & (ye_tr == 1)) / (np.sum(np.exp(np.clip(m_tr[rs], -15, 15))) + 1e-10)
    H0 = np.cumsum(h0); probs = {}
    for h in [12, 24, 48, 72]:
        idx = np.searchsorted(ut, h, side='right') - 1
        probs[f'prob_{h}h'] = 1 - np.exp(-(H0[idx] if idx >= 0 else H0[-1]) * np.exp(np.clip(m.predict(X), -15, 15)))
    return pd.DataFrame(probs)

all_oof = []; all_test = []
for s in range(50):
    kf = KFold(n_splits=5, shuffle=True, random_state=s)
    for tr_idx, val_idx in kf.split(train):
        X_tr, X_val = train.iloc[tr_idx][features], train.iloc[val_idx][features]
        yt_tr, ye_tr = y_time.iloc[tr_idx], y_event.iloc[tr_idx]
        yt_val, ye_val = y_time.iloc[val_idx], y_event.iloc[val_idx]
        
        # Regularized Master Model
        m = xgb.XGBRegressor(objective='survival:cox', n_estimators=45, max_depth=2, reg_lambda=5.0, learning_rate=0.05, random_state=s)
        m.fit(X_tr, y_xgb.iloc[tr_idx])
        
        p_val = get_breslow(m, X_val, X_tr, yt_tr.values, ye_tr.values)
        p_test = get_breslow(m, test[features], X_tr, yt_tr.values, ye_tr.values)
        
        # OOF Calibration (24h+)
        for i, h in enumerate([24, 48, 72]):
            ir = IsotonicRegression(out_of_bounds='clip').fit(p_val[f'prob_{h}h'], ((ye_val == 1) & (yt_val <= h)).astype(int))
            p_val[f'prob_{h}h'] = ir.predict(p_val[f'prob_{h}h'])
            p_test[f'prob_{h}h'] = ir.predict(p_test[f'prob_{h}h'])
            
        all_oof.append(pd.DataFrame(index=val_idx, data=p_val))
        all_test.append(p_test)

final_oof = pd.concat(all_oof).groupby(level=0).mean().fillna(0.01)
final_test = sum(all_test) / len(all_test)

# --- 4. Final Polishing ---
print("Enforcing Monotonicity and clipping...")
for c in ['prob_12h','prob_24h','prob_48h','prob_72h']: final_test[c] = final_test[c].clip(0.001, 0.999)
final_test['prob_24h'] = np.maximum(final_test['prob_24h'], final_test['prob_12h'] + 1e-6)
final_test['prob_48h'] = np.maximum(final_test['prob_48h'], final_test['prob_24h'] + 1e-6)
final_test['prob_72h'] = np.maximum(final_test['prob_72h'], final_test['prob_48h'] + 1e-6)

# --- 5. Validation ---
print("\n--- FINAL VALIDATION ---")
y_time_metric = np.where(y_event == 1, y_time, 72.1)
c_idx = concordance_index(y_time_metric, -final_oof['prob_12h'], y_event)
wb = 0
for h, w in zip([24, 48, 72], [0.3, 0.4, 0.3]):
    mask = (y_event == 1) | (y_time_metric >= h)
    y_true_h = ((y_event == 1) & (y_time_metric <= h)).astype(int)
    wb += w * np.mean((final_oof[f'prob_{h}h'][mask] - y_true_h[mask])**2)
hybrid = 0.3 * c_idx + 0.7 * (1 - wb)

print(f"C-Index (12h): {c_idx:.5f} | Hybrid: {hybrid:.5f}")

sub_path = "submissions/final/submission_specialist_stack_v77.csv"
submission = pd.DataFrame({'event_id': test_raw['event_id']})
for h in [12, 24, 48, 72]: submission[f'prob_{h}h'] = final_test[f'prob_{h}h'].values
submission.to_csv(sub_path, index=False)
print(f"V77 SUCCESS: {sub_path}")
