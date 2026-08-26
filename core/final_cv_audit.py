import pandas as pd
import numpy as np
import xgboost as xgb
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold
import warnings

warnings.filterwarnings('ignore')

def calculate_hybrid(y_time, y_event, probs_df):
    def brier_at(p, h):
        mask = (y_event == 1) | (y_time >= h)
        y_true = ((y_event == 1) & (y_time <= h)).astype(int)
        return np.mean((p[mask] - y_true[mask])**2)
    
    wb = 0.3 * brier_at(probs_df['prob_24h'], 24) + 0.4 * brier_at(probs_df['prob_48h'], 48) + 0.3 * brier_at(probs_df['prob_72h'], 72)
    c_idx = concordance_index(y_time, -probs_df['prob_48h'], y_event)
    return 0.3 * c_idx + 0.7 * (1 - wb)

print("Starting Final CV Audit...")

# --- 1. Load Data ---
train = pd.read_csv("dataset/train.csv")
train['eta_hours'] = train['dist_min_ci_0_5h'] / (train['closing_speed_m_per_h'] * train['alignment_abs'] + 1.0)
y_time, y_event = train['time_to_hit_hours'], train['event']
y_xgb = y_time * np.where(y_event == 1, 1, -1)
features = ['dist_min_ci_0_5h', 'closing_speed_m_per_h', 'alignment_abs', 'area_growth_rate_ha_per_h', 'num_perimeters_0_5h', 'eta_hours', 'area_first_ha', 'centroid_speed_m_per_h', 'dist_slope_ci_0_5h']
X = train[features].fillna(0)

# --- 2. 5-Fold OOF Margins ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_margins = np.zeros(len(train))
for tr_idx, val_idx in kf.split(X):
    m_oof = np.zeros(len(val_idx))
    for s in [42, 123, 456, 789, 10]:
        m = xgb.XGBRegressor(objective='survival:cox', n_estimators=45, max_depth=3, learning_rate=0.05, random_state=s)
        m.fit(X.iloc[tr_idx], y_xgb.iloc[tr_idx])
        m_oof += m.predict(X.iloc[val_idx]) / 5
    oof_margins[val_idx] = m_oof

# --- 3. Evaluate V42 Logic (Heuristic Multipliers) ---
ranks = pd.Series(oof_margins).rank(method='first')
norm_ranks = (ranks - 0.5) / len(ranks)
base_sig = 0.01 + (0.39) * norm_ranks
mults = [1.0, 1.45, 1.95, 2.45]
v42_oof = pd.DataFrame({
    'prob_12h': base_sig * mults[0],
    'prob_24h': base_sig * mults[1],
    'prob_48h': base_sig * mults[2],
    'prob_72h': base_sig * mults[3]
}).clip(0, 0.99)

score_v42 = calculate_hybrid(y_time, y_event, v42_oof)

# --- 4. Evaluate V47 Logic (Empirical Truth) ---
v47_oof = pd.DataFrame()
horizons = [(0.004, 0.57, 12), (0.012, 0.84, 24), (0.025, 0.97, 48), (0.048, 0.985, 72)]
for floor, ceil, h in horizons:
    v47_oof[f'prob_{h}h'] = floor + (ceil - floor) * (norm_ranks ** 1.2)

score_v47 = calculate_hybrid(y_time, y_event, v47_oof)

print(f"\n--- FINAL CV RESULTS ---")
print(f"V42 (Heuristic Absolute): {score_v42:.5f}")
print(f"V47 (Analytical Truth):    {score_v47:.5f}")
