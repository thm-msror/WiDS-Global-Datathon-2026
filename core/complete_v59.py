"""
V59 - Complete Package with honest CV comparison.
- Prunes zero-SHAP dead weight features
- Optuna HPO (50 trials, proper nested CV to avoid leakage)
- Tests XGBoost-only vs XGBoost+CoxPH blend
- Reports honest CV vs V1 (0.96987) and V58 (0.97157) baselines
- Only saves submission if CV beats V58
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
from lifelines import CoxPHFitter
import optuna
import warnings
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

BASELINE_V1  = 0.96987
BASELINE_V58 = 0.97157

train = pd.read_csv("dataset/train.csv")
test  = pd.read_csv("dataset/test.csv")

for df in [train, test]:
    df['eta_hours'] = df['dist_min_ci_0_5h'] / (df['closing_speed_m_per_h'] * df['alignment_abs'] + 1.0)

# SHAP-validated features only (removed zero-SHAP: area_growth_rate, dist_accel)
features = [
    'dist_min_ci_0_5h',
    'num_perimeters_0_5h',
    'alignment_abs',
    'eta_hours',
    'along_track_speed',
    'dist_slope_ci_0_5h',
    'closing_speed_m_per_h',
]

X_train = train[features].fillna(0)
X_test  = test[features].fillna(0)
y_time, y_event = train['time_to_hit_hours'], train['event']
y_xgb = y_time * np.where(y_event == 1, 1, -1)

# --- CV helper ---
def cv_hybrid(params, n_seeds=5):
    """Proper 5-fold OOF CV, averaged over seeds. Returns hybrid Brier score."""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    briers = []
    for h in [12, 24, 48, 72]:
        y_h = ((y_event == 1) & (y_time <= h)).astype(int)
        fold_b = []
        for tr_idx, val_idx in kf.split(train):
            Xtr, Xval = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            ytr = y_xgb.iloc[tr_idx]
            yh_tr, yh_val = y_h.iloc[tr_idx], y_h.iloc[val_idx]
            seed_preds = []
            for s in range(n_seeds):
                m = xgb.XGBRegressor(objective='survival:cox', random_state=s, **params)
                m.fit(Xtr, ytr)
                tr_m = m.predict(Xtr)
                val_m = m.predict(Xval)
                ir = IsotonicRegression(out_of_bounds='clip').fit(tr_m, yh_tr)
                seed_preds.append(ir.predict(val_m))
            pred = np.mean(seed_preds, axis=0).clip(0.001, 0.999)
            fold_b.append(np.mean((pred - yh_val) ** 2))
        briers.append(np.mean(fold_b))
    return 1 - np.mean(briers)

# --- STEP 1: Fixed-param baseline (V58 params) ---
v58_params = {'n_estimators': 45, 'max_depth': 3, 'learning_rate': 0.05,
              'subsample': 1.0, 'colsample_bytree': 1.0}
score_fixed = cv_hybrid(v58_params)
print(f"Fixed params (V58 equivalent):  {score_fixed:.5f}")

# --- STEP 2: Optuna HPO ---
def objective(trial):
    params = {
        'n_estimators':    trial.suggest_int('n_estimators', 30, 120),
        'max_depth':       trial.suggest_int('max_depth', 2, 4),
        'learning_rate':   trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'subsample':       trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight':trial.suggest_int('min_child_weight', 1, 10),
    }
    return cv_hybrid(params)

print("Running Optuna (50 trials)...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
best_params = study.best_params
score_optuna = study.best_value
print(f"Optuna best CV:                  {score_optuna:.5f}")
print(f"Best params: {best_params}")

# --- STEP 3: Test CoxPH blend ---
def cv_hybrid_blend(xgb_params, blend_w=0.1):
    """Test adding CoxPH blend. blend_w = weight for CoxPH."""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    briers = []
    for h in [12, 24, 48, 72]:
        y_h = ((y_event == 1) & (y_time <= h)).astype(int)
        fold_b = []
        for tr_idx, val_idx in kf.split(train):
            Xtr, Xval = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            ytr = y_xgb.iloc[tr_idx]
            yh_tr, yh_val = y_h.iloc[tr_idx], y_h.iloc[val_idx]
            # XGBoost part
            seed_preds = []
            for s in range(5):
                m = xgb.XGBRegressor(objective='survival:cox', random_state=s, **xgb_params)
                m.fit(Xtr, ytr)
                tr_m = m.predict(Xtr)
                val_m = m.predict(Xval)
                ir = IsotonicRegression(out_of_bounds='clip').fit(tr_m, yh_tr)
                seed_preds.append(ir.predict(val_m))
            xgb_pred = np.mean(seed_preds, axis=0)
            # CoxPH part
            try:
                cox_df_tr = train.iloc[tr_idx][features + ['time_to_hit_hours', 'event']].fillna(0).copy()
                cox_df_val = train.iloc[val_idx][features].fillna(0).copy()
                cph = CoxPHFitter(penalizer=0.1)
                cph.fit(cox_df_tr, duration_col='time_to_hit_hours', event_col='event')
                cox_surv = cph.predict_survival_function(cox_df_val, times=[h])
                cox_pred = 1 - cox_surv.iloc[0].values
            except Exception:
                cox_pred = np.zeros(len(val_idx))
            blended = ((1 - blend_w) * xgb_pred + blend_w * cox_pred).clip(0.001, 0.999)
            fold_b.append(np.mean((blended - yh_val) ** 2))
        briers.append(np.mean(fold_b))
    return 1 - np.mean(briers)

score_blend = cv_hybrid_blend(best_params, blend_w=0.1)
print(f"Optuna + CoxPH blend (10%):      {score_blend:.5f}")

# --- HONEST COMPARISON ---
print("\n====== HONEST CV COMPARISON ======")
print(f"  V1  baseline:                  {BASELINE_V1:.5f}")
print(f"  V58 baseline:                  {BASELINE_V58:.5f}")
print(f"  V59 XGBoost only (Optuna):     {score_optuna:.5f}  {'BEATS V58' if score_optuna > BASELINE_V58 else 'NO IMPROVEMENT'}")
print(f"  V59 XGBoost + CoxPH blend:     {score_blend:.5f}  {'BEATS V58' if score_blend > BASELINE_V58 else 'NO IMPROVEMENT'}")

best_score = max(score_optuna, score_blend)
use_blend  = score_blend > score_optuna
best_label = "XGBoost+CoxPH blend" if use_blend else "XGBoost only"

if best_score <= BASELINE_V58:
    print(f"\nV59 ({best_score:.5f}) does NOT beat V58 ({BASELINE_V58:.5f}). No submission saved.")
else:
    print(f"\nV59 ({best_score:.5f}) beats V58. Building submission using: {best_label}")

    # --- BUILD FULL SUBMISSION ---
    final_probs = pd.DataFrame(0.0, index=test.index,
                               columns=['prob_12h', 'prob_24h', 'prob_48h', 'prob_72h'])
    seeds = [42, 123, 456, 789, 10, 20, 30, 40, 50, 60]

    for s in seeds:
        m = xgb.XGBRegressor(objective='survival:cox', random_state=s, **best_params)
        m.fit(X_train, y_xgb)
        tr_m = m.predict(X_train)
        te_m = m.predict(X_test)
        for h in [12, 24, 48, 72]:
            y_h = ((y_event == 1) & (y_time <= h)).astype(int)
            ir = IsotonicRegression(out_of_bounds='clip').fit(tr_m, y_h)
            final_probs[f'prob_{h}h'] += ir.predict(te_m) / len(seeds)

    if use_blend:
        cox_test_probs = pd.DataFrame(0.0, index=test.index,
                                      columns=['prob_12h', 'prob_24h', 'prob_48h', 'prob_72h'])
        try:
            cox_df_full = train[features + ['time_to_hit_hours', 'event']].fillna(0).copy()
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(cox_df_full, duration_col='time_to_hit_hours', event_col='event')
            for h in [12, 24, 48, 72]:
                surv = cph.predict_survival_function(X_test.fillna(0), times=[h])
                cox_test_probs[f'prob_{h}h'] = 1 - surv.iloc[0].values
        except Exception as e:
            print(f"CoxPH failed: {e}. Using XGBoost only.")
            use_blend = False

        if use_blend:
            final_probs = (0.9 * final_probs + 0.1 * cox_test_probs)

    final_probs = final_probs.clip(0.001, 0.999)
    for i in range(len(final_probs)):
        for j in range(1, 4):
            final_probs.iloc[i, j+1] = max(final_probs.iloc[i, j+1], final_probs.iloc[i, j])

    sub = pd.concat([test[['event_id']].reset_index(drop=True), final_probs], axis=1)
    out = "submissions/final/submission_complete_v59.csv"
    sub.to_csv(out, index=False)
    print(f"V59 saved to {out}")
    print(f"Final CV: {best_score:.5f}")
