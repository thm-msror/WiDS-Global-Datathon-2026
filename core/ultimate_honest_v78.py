import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression
from lifelines.utils import concordance_index
import warnings

warnings.filterwarnings('ignore')

print("Executing V78 'THE HIGH-DIMENSIONAL HONEST ENSEMBLE' (Final Boss Build)...")

# --- 1. Data Loading and Specialist Meta-Feature ---
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

# Specialist Score (The Ranking DNA)
print("Injecting Specialist DNA...")
hits_idx = train[train['event'] == 1].index
spec_feats = ['closing_speed_m_per_h', 'alignment_abs', 'area_growth_rate_ha_per_h', 'centroid_speed_m_per_h', 'dist_min_ci_0_5h', 'along_track_speed']
m_spec = xgb.XGBRegressor(n_estimators=50, max_depth=2, random_state=42)
m_spec.fit(train.loc[hits_idx][spec_feats], train.loc[hits_idx]['time_to_hit_hours'])
train['spec_score'] = m_spec.predict(train[spec_feats])
test['spec_score'] = m_spec.predict(test[spec_feats])

features = [
    'dist_min_ci_0_5h', 'log_dist', 'num_perimeters_0_5h', 
    'alignment_abs', 'threat_alignment', 'growth_momentum', 
    'is_extreme_close', 'area_growth_rate_ha_per_h', 'closing_speed_m_per_h',
    'dt_first_last_0_5h', 'area_first_ha', 'centroid_speed_m_per_h',
    'spec_score', 'along_track_speed'
]

# --- 2. Master Ensemble (50 Seeds x 5 Folds = 250 Models) ---
print(f"Training 250 Models with Honest OOF Calibration...")
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

all_test_cal = []
for s in range(50):
    if s % 10 == 0: print(f"Batch {s}...")
    kf = KFold(n_splits=5, shuffle=True, random_state=s) # V64 Secret: Unique Folds per Seed
    for tr_idx, val_idx in kf.split(train):
        X_tr, X_val = train.iloc[tr_idx][features], train.iloc[val_idx][features]
        yt_tr, ye_tr = y_time.iloc[tr_idx], y_event.iloc[tr_idx]
        yt_val, ye_val = y_time.iloc[val_idx], y_event.iloc[val_idx]
        
        m = xgb.XGBRegressor(objective='survival:cox', n_estimators=45, max_depth=2, reg_lambda=10.0, learning_rate=0.05, random_state=s)
        m.fit(X_tr, y_xgb.iloc[tr_idx])
        
        p_val = get_breslow(m, X_val, X_tr, yt_tr.values, ye_tr.values)
        p_test = get_breslow(m, test[features], X_tr, yt_tr.values, ye_tr.values)
        
        # Honest OOF Calibration
        p_cal_test = p_test.copy()
        for i, h in enumerate([12, 24, 48, 72]):
            y_h_val = ((ye_val == 1) & (yt_val <= h)).astype(int)
            ir = IsotonicRegression(out_of_bounds='clip').fit(p_val.iloc[:, i], y_h_val)
            p_cal_test.iloc[:, i] = ir.predict(p_test.iloc[:, i])
        all_test_cal.append(p_cal_test)

final_test = sum(all_test_cal) / len(all_test_cal)

# --- 3. Final Polishing ---
# Monotonicity
for i in range(len(final_test)):
    final_test.iloc[i, 1] = max(final_test.iloc[i, 1], final_test.iloc[i, 0] + 1e-6)
    final_test.iloc[i, 2] = max(final_test.iloc[i, 2], final_test.iloc[i, 1] + 1e-6)
    final_test.iloc[i, 3] = max(final_test.iloc[i, 3], final_test.iloc[i, 2] + 1e-6)

final_test = final_test.clip(0.001, 0.999)

sub_path = "submissions/final/submission_ultimate_honest_v78.csv"
submission = pd.DataFrame({'event_id': test_raw['event_id']})
for i, h in enumerate([12, 24, 48, 72]):
    submission[f'prob_{h}h'] = final_test.iloc[:, i].values
submission.to_csv(sub_path, index=False)

print(f"\nV78 SUCCESS: {sub_path}")
print(submission.head())
