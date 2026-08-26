import pandas as pd
import numpy as np
import xgboost as xgb
from lifelines import CoxPHFitter
from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression
import warnings

warnings.filterwarnings('ignore')

# --- 1. Extreme Feature Pruning (The "Essential 8") ---
def engineer_features(df):
    df = df.copy()
    df['growth_momentum'] = df['area_growth_rate_ha_per_h'] * df['dt_first_last_0_5h']
    df['threat_alignment'] = df['closing_speed_m_per_h'] * df['alignment_abs']
    df['log_dist'] = np.log1p(df['dist_min_ci_0_5h'])
    
    # ONLY the most robust physical signals
    features = [
        'dist_min_ci_0_5h', 'log_dist', 
        'alignment_abs', 'threat_alignment', 'growth_momentum', 
        'area_growth_rate_ha_per_h', 'closing_speed_m_per_h',
        'dt_first_last_0_5h'
    ]
    return df[features + (['time_to_hit_hours', 'event'] if 'event' in df.columns else [])].fillna(0)

def get_breslow_probs(m_target, m_tr, times_tr, events_tr, horizons=[12, 24, 48, 72]):
    unique_times = np.sort(np.unique(times_tr))
    h0 = np.zeros_like(unique_times)
    for i, t in enumerate(unique_times):
        risk_set = times_tr >= t
        h0[i] = np.sum((times_tr == t) & (events_tr == 1)) / (np.sum(np.exp(np.clip(m_tr[risk_set], -15, 15))) + 1e-10)
    H0 = np.cumsum(h0)
    probs = {}
    for h in horizons:
        idx = np.searchsorted(unique_times, h, side='right') - 1
        H0_h = H0[idx] if idx >= 0 else 0
        p = 1 - np.exp(-H0_h * np.exp(np.clip(m_target, -15, 15)))
        probs[f'prob_{h}h'] = np.nan_to_num(p, nan=0.0)
    return pd.DataFrame(probs)

# --- 2. The 50-Seed Pure Physics Engine ---
print("Building 50-Seed Pure Physics Engine (Maximum Stability)...")
train_raw = pd.read_csv("dataset/train.csv")
test_raw = pd.read_csv("dataset/test.csv")

X_train = engineer_features(train_raw)
X_test = engineer_features(test_raw)
features = [f for f in X_train.columns if f not in ['time_to_hit_hours', 'event']]

# 50 Seeds for extreme variance reduction
seeds = list(range(50))
all_test_probs = []

for seed in seeds:
    if seed % 10 == 0: print(f"Processing Batch {seed}-{seed+9}...")
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    seed_probs = []

    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr_time, y_tr_event = train_raw.iloc[train_idx]['time_to_hit_hours'], train_raw.iloc[train_idx]['event']
        
        # Base XGB Ranker
        m = xgb.XGBRegressor(objective='survival:cox', n_estimators=40, max_depth=3, learning_rate=0.05, random_state=seed)
        m.fit(X_tr[features], y_tr_time * np.where(y_tr_event == 1, 1, -1))
        
        # Raw Breslow Probs (Smooth)
        p_raw = get_breslow_probs(m.predict(X_test[features]), m.predict(X_tr[features]), y_tr_time.values, y_tr_event.values)
        
        # Partial Isotonic Calibration (Softened)
        p_cal = p_raw.copy()
        for h in [12, 24, 48, 72]:
            y_h = ((y_tr_event == 1) & (y_tr_time <= h)).astype(int)
            p_tr = get_breslow_probs(m.predict(X_tr[features]), m.predict(X_tr[features]), y_tr_time.values, y_tr_event.values)[f'prob_{h}h']
            ir = IsotonicRegression(out_of_bounds='clip').fit(p_tr, y_h)
            # 80% Smooth / 20% Isotonic for stability
            p_cal[f'prob_{h}h'] = 0.8 * p_raw[f'prob_{h}h'] + 0.2 * ir.predict(p_raw[f'prob_{h}h'])
            
        seed_probs.append(p_cal)
    all_test_probs.append(sum(seed_probs) / 5)

final_test = sum(all_test_probs) / 50

# Monotonicity
final_test['prob_24h'] = np.maximum(final_test['prob_24h'], final_test['prob_12h'])
final_test['prob_48h'] = np.maximum(final_test['prob_48h'], final_test['prob_24h'])
final_test['prob_72h'] = np.maximum(final_test['prob_72h'], final_test['prob_48h'])

sub_path = "submissions/final/submission_pure_physics_50_seed_v5.csv"
submission = pd.DataFrame({'event_id': test_raw['event_id']})
for h in [12, 24, 48, 72]: submission[f'prob_{h}h'] = final_test[f'prob_{h}h'].values
submission.to_csv(sub_path, index=False)

print(f"PURE PHYSICS V5 saved to {sub_path}")
