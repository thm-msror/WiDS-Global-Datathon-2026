import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression
from lifelines.utils import concordance_index
import os
import warnings
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from log_run import log_run

warnings.filterwarnings('ignore')

# --- 1. Features ---
def engineer_features(df):
    df = df.copy()
    df['growth_momentum'] = df['area_growth_rate_ha_per_h'] * df['dt_first_last_0_5h']
    df['threat_alignment'] = df['closing_speed_m_per_h'] * df['alignment_abs']
    df['is_extreme_close'] = (df['dist_min_ci_0_5h'] < 5000).astype(int)
    df['log_dist'] = np.log1p(df['dist_min_ci_0_5h'])
    features = [
        'dist_min_ci_0_5h', 'log_dist', 'num_perimeters_0_5h', 
        'alignment_abs', 'threat_alignment', 'growth_momentum', 
        'is_extreme_close', 'area_growth_rate_ha_per_h', 'closing_speed_m_per_h',
        'dt_first_last_0_5h', 'area_first_ha', 'centroid_speed_m_per_h'
    ]
    return df[features + (['time_to_hit_hours', 'event'] if 'event' in df.columns else [])].fillna(0)

def get_breslow_probs(margins_target, margins_tr, times_tr, events_tr, index, horizons=[12, 24, 48, 72]):
    unique_times = np.sort(np.unique(times_tr))
    h0 = np.zeros_like(unique_times)
    for i, t in enumerate(unique_times):
        risk_set = times_tr >= t
        h0[i] = np.sum((times_tr == t) & (events_tr == 1)) / (np.sum(np.exp(np.clip(margins_tr[risk_set], -15, 15))) + 1e-10)
    H0 = np.cumsum(h0)
    probs = {}
    for h in horizons:
        idx = np.searchsorted(unique_times, h, side='right') - 1
        H0_h = H0[idx] if idx >= 0 else 0
        p = 1 - np.exp(-H0_h * np.exp(np.clip(margins_target, -15, 15)))
        probs[f'prob_{h}h'] = np.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0)
    return pd.DataFrame(probs, index=index)

def brier_score_at_horizon(prob_pred, event_true, time_true, horizon):
    mask = (event_true == 1) | (time_true >= horizon)
    y_true = (event_true == 1) & (time_true <= horizon)
    if np.sum(mask) == 0: return 0
    return np.mean((prob_pred[mask] - y_true[mask])**2)

# --- 2. Execution ---
print("Running Isotonic Calibration on XGBoost Champion...")
train_raw = pd.read_csv("dataset/train.csv")
test_raw = pd.read_csv("dataset/test.csv")

X_train_full = engineer_features(train_raw)
X_test_full = engineer_features(test_raw)
features = [f for f in X_train_full.columns if f not in ['time_to_hit_hours', 'event']]

y_surv = train_raw[['time_to_hit_hours', 'event']]
y_xgb = y_surv['time_to_hit_hours'] * np.where(y_surv['event'] == 1, 1, -1)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
horizons = [12, 24, 48, 72]

# Step A: Get OOF Predictions (Breslow base)
oof_base_probs = []
test_base_probs = []

for train_idx, val_idx in kf.split(X_train_full):
    X_tr, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
    y_tr_xgb = y_xgb.iloc[train_idx]
    
    m = xgb.XGBRegressor(objective='survival:cox', n_estimators=45, max_depth=3, learning_rate=0.05, random_state=42)
    m.fit(X_tr[features], y_tr_xgb)
    m_tr_preds = m.predict(X_tr[features])
    
    p_val = get_breslow_probs(m.predict(X_val[features]), m_tr_preds, X_tr['time_to_hit_hours'].values, X_tr['event'].values, index=X_val.index)
    p_test = get_breslow_probs(m.predict(X_test_full[features]), m_tr_preds, X_tr['time_to_hit_hours'].values, X_tr['event'].values, index=test_raw.index)
    
    oof_base_probs.append(p_val)
    test_base_probs.append(p_test)

final_oof_base = pd.concat(oof_base_probs).sort_index()
final_test_base = sum(test_base_probs) / 5

# Step B: Apply Isotonic Calibration to the OOF results
print("Performing Isotonic Calibration per horizon...")
final_oof_calibrated = final_oof_base.copy()
final_test_calibrated = final_test_base.copy()

for h in horizons:
    y_h = ((train_raw['event'] == 1) & (train_raw['time_to_hit_hours'] <= h)).astype(int)
    
    # Fit Isotonic Regression on OOF probabilities
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(final_oof_base[f'prob_{h}h'], y_h)
    
    # Transform OOF and Test
    final_oof_calibrated[f'prob_{h}h'] = ir.predict(final_oof_base[f'prob_{h}h'])
    final_test_calibrated[f'prob_{h}h'] = ir.predict(final_test_base[f'prob_{h}h'])

# Monotonicity
final_test_calibrated['prob_24h'] = np.maximum(final_test_calibrated['prob_24h'], final_test_calibrated['prob_12h'])
final_test_calibrated['prob_48h'] = np.maximum(final_test_calibrated['prob_48h'], final_test_calibrated['prob_24h'])
final_test_calibrated['prob_72h'] = np.maximum(final_test_calibrated['prob_72h'], final_test_calibrated['prob_48h'])

# Metrics
y_true = train_raw.loc[final_oof_calibrated.index]
b24 = brier_score_at_horizon(final_oof_calibrated['prob_24h'].values, y_true['event'].values, y_true['time_to_hit_hours'].values, 24)
b48 = brier_score_at_horizon(final_oof_calibrated['prob_48h'].values, y_true['event'].values, y_true['time_to_hit_hours'].values, 48)
b72 = brier_score_at_horizon(final_oof_calibrated['prob_72h'].values, y_true['event'].values, y_true['time_to_hit_hours'].values, 72)
wb = 0.3 * b24 + 0.4 * b48 + 0.3 * b72
c_idx = concordance_index(y_true['time_to_hit_hours'].values, -final_oof_calibrated['prob_48h'].values, y_true['event'].values)
hybrid = 0.3 * c_idx + 0.7 * (1 - wb)

print(f"\n--- ISOTONIC CALIBRATED CV: {hybrid:.4f} ---")

# Save
submission = pd.DataFrame({'event_id': test_raw['event_id']})
for h in [12, 24, 48, 72]: submission[f'prob_{h}h'] = final_test_calibrated[f'prob_{h}h'].values
submission.to_csv("submissions/submission_isotonic_champion.csv", index=False)

log_run("Isotonic Calibrated XGBoost", {'Hybrid Score': hybrid})
