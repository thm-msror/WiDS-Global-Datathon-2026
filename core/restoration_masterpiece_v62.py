import pandas as pd
import numpy as np
import xgboost as xgb
from lifelines import CoxPHFitter
from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression
import warnings

warnings.filterwarnings('ignore')

print("Executing V62 'The Restoration Masterpiece' (Restoring V1/V10 DNA)...")

# --- 1. The Anchor Feature Set (12 Features) ---
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
    return df[features].fillna(0)

train_raw = pd.read_csv("dataset/train.csv")
test_raw = pd.read_csv("dataset/test.csv")

X_train = engineer_features(train_raw)
X_test = engineer_features(test_raw)
y_time, y_event = train_raw['time_to_hit_hours'], train_raw['event']
y_xgb = y_time * np.where(y_event == 1, 1, -1)

# --- 2. The Breslow Engine ---
def get_breslow_probs(margins_target, margins_tr, times_tr, events_tr, horizons=[12, 24, 48, 72]):
    unique_times = np.sort(np.unique(times_tr))
    h0 = np.zeros_like(unique_times)
    for i, t in enumerate(unique_times):
        risk_set = times_tr >= t
        h0[i] = np.sum((times_tr == t) & (events_tr == 1)) / (np.sum(np.exp(np.clip(margins_tr[risk_set], -15, 15))) + 1e-10)
    H0 = np.cumsum(h0)
    probs = {}
    for h in horizons:
        idx = np.searchsorted(unique_times, h, side='right') - 1
        H0_h = H0[idx] if idx >= 0 else H0[-1]
        p = 1 - np.exp(-H0_h * np.exp(np.clip(margins_target, -15, 15)))
        probs[f'prob_{h}h'] = np.nan_to_num(p, nan=0.0)
    return pd.DataFrame(probs)

# --- 3. Stable Champion Parameters (No Optuna Overfit) ---
seeds = [42, 123, 456, 789, 10, 20, 30, 40, 50, 60]
all_test_cal = []

for s in seeds:
    print(f"Training Seed {s}...")
    # Ranking Head
    m_xgb = xgb.XGBRegressor(objective='survival:cox', n_estimators=45, max_depth=3, learning_rate=0.05, random_state=s)
    m_xgb.fit(X_train, y_xgb)
    
    # Generate Breslow Probs
    p_xgb_test = get_breslow_probs(m_xgb.predict(X_test), m_xgb.predict(X_train), y_time.values, y_event.values)
    
    # Apply Isotonic Calibration (using the whole train set as it is the final build)
    p_calibrated_test = p_xgb_test.copy()
    for i, h in enumerate([12, 24, 48, 72]):
        y_h = ((y_event == 1) & (y_time <= h)).astype(int)
        p_tr_xgb = get_breslow_probs(m_xgb.predict(X_train), m_xgb.predict(X_train), y_time.values, y_event.values)[f'prob_{h}h']
        ir = IsotonicRegression(out_of_bounds='clip').fit(p_tr_xgb, y_h)
        p_calibrated_test[f'prob_{h}h'] = ir.predict(p_xgb_test[f'prob_{h}h'])
    
    all_test_cal.append(p_calibrated_test)

# Final Ensemble
final_test = sum(all_test_cal) / len(seeds)

# Monotonicity Fix
for i in range(len(final_test)):
    final_test.iloc[i, 1] = max(final_test.iloc[i, 1], final_test.iloc[i, 0])
    final_test.iloc[i, 2] = max(final_test.iloc[i, 2], final_test.iloc[i, 1])
    final_test.iloc[i, 3] = max(final_test.iloc[i, 3], final_test.iloc[i, 2])

# Defensive Clipping (0.001 - 0.999)
final_test = final_test.clip(0.001, 0.999)

# Save Submission
sub_path = "submissions/final/submission_restoration_masterpiece_v62.csv"
submission = pd.DataFrame({'event_id': test_raw['event_id']})
for h in [12, 24, 48, 72]:
    submission[f'prob_{h}h'] = final_test[f'prob_{h}h'].values
submission.to_csv(sub_path, index=False)

print(f"V62 RESTORATION saved to {sub_path}")
