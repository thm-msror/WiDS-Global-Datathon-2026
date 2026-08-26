"""
V81: V64 at 50 seeds — the highest-OOF model rebuilt with maximum stability
Same architecture as V64 (honest isotonic all 4 horizons, unique folds per seed)
Same features/hyperparams as V1
Just 50 seeds instead of 20 → 250 models instead of 100
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression
import warnings
warnings.filterwarnings('ignore')

print("V81: V64 at 50 Seeds (The 0.97 Attempt)...")

def engineer_features(df):
    df = df.copy()
    df['growth_momentum']  = df['area_growth_rate_ha_per_h'] * df['dt_first_last_0_5h']
    df['threat_alignment'] = df['closing_speed_m_per_h'] * df['alignment_abs']
    df['is_extreme_close'] = (df['dist_min_ci_0_5h'] < 5000).astype(int)
    df['log_dist']         = np.log1p(df['dist_min_ci_0_5h'])
    return df[[
        'dist_min_ci_0_5h', 'log_dist', 'num_perimeters_0_5h',
        'alignment_abs', 'threat_alignment', 'growth_momentum',
        'is_extreme_close', 'area_growth_rate_ha_per_h', 'closing_speed_m_per_h',
        'dt_first_last_0_5h', 'area_first_ha', 'centroid_speed_m_per_h'
    ]].fillna(0)

train_raw = pd.read_csv("dataset/train.csv")
test_raw  = pd.read_csv("dataset/test.csv")
X_train   = engineer_features(train_raw)
X_test    = engineer_features(test_raw)
y_time    = train_raw['time_to_hit_hours']
y_event   = train_raw['event']
y_xgb     = y_time * np.where(y_event == 1, 1, -1)

def get_breslow(m, X_pred, X_tr, yt_tr, ye_tr):
    m_tr = m.predict(X_tr)
    ut   = np.sort(np.unique(yt_tr)); h0 = np.zeros_like(ut)
    for i, t in enumerate(ut):
        rs    = yt_tr >= t
        h0[i] = np.sum((yt_tr == t) & (ye_tr == 1)) / \
                (np.sum(np.exp(np.clip(m_tr[rs], -15, 15))) + 1e-10)
    H0 = np.cumsum(h0); out = {}
    for h in [12, 24, 48, 72]:
        idx  = np.searchsorted(ut, h, side='right') - 1
        out[f'prob_{h}h'] = 1 - np.exp(-(H0[idx] if idx >= 0 else H0[-1]) *
                            np.exp(np.clip(m.predict(X_pred), -15, 15)))
    return pd.DataFrame(out)

# 250 models: 50 seeds × 5 unique folds (exact V64 formula, 2.5x scale)
all_test_cal = []
for s in range(50):
    if s % 10 == 0: print(f"  Seed {s}/50...")
    kf = KFold(n_splits=5, shuffle=True, random_state=s)  # unique folds per seed
    for tr_idx, val_idx in kf.split(X_train):
        X_tr, X_val   = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        yt_tr, ye_tr  = y_time.iloc[tr_idx],  y_event.iloc[tr_idx]
        yt_val, ye_val = y_time.iloc[val_idx], y_event.iloc[val_idx]

        m = xgb.XGBRegressor(objective='survival:cox', n_estimators=45,
                             max_depth=3, learning_rate=0.05, random_state=s)
        m.fit(X_tr, y_xgb.iloc[tr_idx])

        p_val  = get_breslow(m, X_val,  X_tr, yt_tr.values, ye_tr.values)
        p_test = get_breslow(m, X_test, X_tr, yt_tr.values, ye_tr.values)

        # Honest OOF isotonic on all 4 horizons (exact V64 formula)
        p_cal = p_test.copy()
        for i, h in enumerate([12, 24, 48, 72]):
            y_h_val = ((ye_val == 1) & (yt_val <= h)).astype(int)
            ir = IsotonicRegression(out_of_bounds='clip').fit(p_val.iloc[:, i], y_h_val)
            p_cal.iloc[:, i] = ir.predict(p_test.iloc[:, i])
        all_test_cal.append(p_cal)

final_test = sum(all_test_cal) / len(all_test_cal)

# Monotonicity
for i in range(len(final_test)):
    final_test.iloc[i, 1] = max(final_test.iloc[i, 1], final_test.iloc[i, 0])
    final_test.iloc[i, 2] = max(final_test.iloc[i, 2], final_test.iloc[i, 1])
    final_test.iloc[i, 3] = max(final_test.iloc[i, 3], final_test.iloc[i, 2])
final_test = final_test.clip(0.001, 0.999)

sub_path = "submissions/final/submission_v81_honest_50seed.csv"
submission = pd.DataFrame({'event_id': test_raw['event_id']})
for i, h in enumerate([12, 24, 48, 72]):
    submission[f'prob_{h}h'] = final_test.iloc[:, i].values
submission.to_csv(sub_path, index=False)

print(f"\nV81 saved: {sub_path}")
print(f"Safe fires at floor: {(submission.prob_12h <= 0.001).sum()}/95")
print(f"prob_12h range: [{submission.prob_12h.min():.4f}, {submission.prob_12h.max():.4f}]")
print(submission.head(15).to_string())
