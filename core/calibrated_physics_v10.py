import pandas as pd
import numpy as np
import xgboost as xgb
from lifelines import CoxPHFitter
from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression
from lifelines.utils import concordance_index
import optuna
import warnings

warnings.filterwarnings('ignore')

print("Building V10 Super-Champion (Fixed Logic)...")

# --- 1. Data and Features (Identical to V1) ---
train_raw = pd.read_csv("dataset/train.csv")
test_raw = pd.read_csv("dataset/test.csv")

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

X_train = engineer_features(train_raw)
X_test = engineer_features(test_raw)
y_time, y_event = train_raw['time_to_hit_hours'], train_raw['event']
y_xgb = y_time * np.where(y_event == 1, 1, -1)

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

# --- 2. Heavy-Duty Bagging (100-Model Average) ---
all_oof_rank = []
all_test_rank = []
all_oof_cal = []
all_test_cal = []

seeds = list(range(20)) # 100 models total

for s in seeds:
    if s % 5 == 0: print(f"Processing Batch {s}...")
    kf = KFold(n_splits=5, shuffle=True, random_state=s)
    oof_rank = pd.DataFrame(index=train_raw.index, columns=['prob_12h','prob_24h','prob_48h','prob_72h'])
    oof_cal = pd.DataFrame(index=train_raw.index, columns=['prob_12h','prob_24h','prob_48h','prob_72h'])
    
    for tr_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr_time, y_tr_event = y_time.iloc[tr_idx], y_event.iloc[tr_idx]
        y_val_time, y_val_event = y_time.iloc[val_idx], y_event.iloc[val_idx]
        
        # A. Ranking Head
        m_xgb = xgb.XGBRegressor(objective='survival:cox', n_estimators=45, max_depth=3, learning_rate=0.05, random_state=s)
        m_xgb.fit(X_tr, y_xgb.iloc[tr_idx])
        p_xgb_val = get_breslow_probs(m_xgb.predict(X_val), m_xgb.predict(X_tr), y_tr_time.values, y_tr_event.values)
        p_xgb_test = get_breslow_probs(m_xgb.predict(X_test), m_xgb.predict(X_tr), y_tr_time.values, y_tr_event.values)
        
        cox_cols = ['log_dist', 'threat_alignment', 'is_extreme_close']
        df_cox = X_tr[cox_cols].copy(); df_cox['time']=y_tr_time; df_cox['event']=y_tr_event
        m_cox = CoxPHFitter(penalizer=0.1).fit(df_cox, duration_col='time', event_col='event')
        p_cox_val = pd.DataFrame({f'prob_{h}h': 1 - m_cox.predict_survival_function(X_val[cox_cols]).iloc[np.abs(m_cox.predict_survival_function(X_val[cox_cols]).index - h).argmin()].values for h in [12, 24, 48, 72]})
        p_cox_test = pd.DataFrame({f'prob_{h}h': 1 - m_cox.predict_survival_function(X_test[cox_cols]).iloc[np.abs(m_cox.predict_survival_function(X_test[cox_cols]).index - h).argmin()].values for h in [12, 24, 48, 72]})
        
        p_rank_val = 0.9 * p_xgb_val.values + 0.1 * p_cox_val.values
        p_rank_test = 0.9 * p_xgb_test.values + 0.1 * p_cox_test.values
        oof_rank.iloc[val_idx] = p_rank_val
        
        # B. Calibration Head (Isotonic on OOF)
        p_cal_val = np.zeros_like(p_rank_val)
        p_cal_test = np.zeros_like(p_rank_test)
        for i, h in enumerate([12, 24, 48, 72]):
            y_h_val = ((y_val_event == 1) & (y_val_time <= h)).astype(int)
            ir = IsotonicRegression(out_of_bounds='clip').fit(p_rank_val[:, i], y_h_val)
            p_cal_val[:, i] = ir.predict(p_rank_val[:, i])
            p_cal_test[:, i] = ir.predict(p_rank_test[:, i])
        oof_cal.iloc[val_idx] = p_cal_val
        
        all_test_rank.append(pd.DataFrame(p_rank_test, columns=['prob_12h','prob_24h','prob_48h','prob_72h']))
        all_test_cal.append(pd.DataFrame(p_cal_test, columns=['prob_12h','prob_24h','prob_48h','prob_72h']))
        
    all_oof_rank.append(oof_rank.astype(float))
    all_oof_cal.append(oof_cal.astype(float))

final_oof_rank = sum(all_oof_rank) / len(seeds)
final_test_rank = sum(all_test_rank) / (len(seeds) * 5)
final_oof_cal = sum(all_oof_cal) / len(seeds)
final_test_cal = sum(all_test_cal) / (len(seeds) * 5)

# --- 3. Optuna Blending ---
def objective(trial):
    w = trial.suggest_float('w', 0.1, 0.9)
    blend_oof = w * final_oof_rank + (1-w) * final_oof_cal
    wb = 0
    for h, weight in zip([24, 48, 72], [0.3, 0.4, 0.3]):
        mask = (y_event == 1) | (y_time >= h)
        y_true_h = ((y_event == 1) & (y_time <= h)).astype(int)
        wb += weight * np.mean((blend_oof[f'prob_{h}h'][mask] - y_true_h[mask])**2)
    c_idx = concordance_index(y_time, -blend_oof['prob_48h'], y_event)
    return 0.3 * c_idx + 0.7 * (1 - wb)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
best_w = study.best_params['w']
print(f"Optimal Blend: {best_w:.4f} Ranking / {1-best_w:.4f} Calibration")

# --- 4. Final Submission ---
final_sub_probs = best_w * final_test_rank + (1-best_w) * final_test_cal
for i in range(len(final_sub_probs)):
    final_sub_probs.iloc[i, 1] = max(final_sub_probs.iloc[i, 1], final_sub_probs.iloc[i, 0])
    final_sub_probs.iloc[i, 2] = max(final_sub_probs.iloc[i, 2], final_sub_probs.iloc[i, 1])
    final_sub_probs.iloc[i, 3] = max(final_sub_probs.iloc[i, 3], final_sub_probs.iloc[i, 2])

sub_path = "submissions/final/submission_calibrated_physics_v10.csv"
submission = pd.DataFrame({'event_id': test_raw['event_id']})
for i, h in enumerate([12, 24, 48, 72]): submission[f'prob_{h}h'] = final_sub_probs.iloc[:, i].values
submission.to_csv(sub_path, index=False)
print(f"V10 SUPER-CHAMPION saved to {sub_path} (CV: {study.best_value:.5f})")
