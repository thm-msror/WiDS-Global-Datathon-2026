import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv("dataset/train.csv")
for df in [train]:
    df['eta_hours'] = df['dist_min_ci_0_5h'] / (df['closing_speed_m_per_h'] * df['alignment_abs'] + 1.0)

features = [
    'dist_min_ci_0_5h', 'closing_speed_m_per_h', 'alignment_abs',
    'area_growth_rate_ha_per_h', 'num_perimeters_0_5h',
    'along_track_speed', 'dist_accel_m_per_h2', 'dist_slope_ci_0_5h', 'eta_hours',
]

feature_labels = {
    'dist_min_ci_0_5h':       'Distance to Line (min)',
    'closing_speed_m_per_h':  'Closing Speed',
    'alignment_abs':          'Alignment (abs)',
    'area_growth_rate_ha_per_h': 'Area Growth Rate',
    'num_perimeters_0_5h':    'Perimeter Count',
    'along_track_speed':      'Along-Track Speed ← NEW',
    'dist_accel_m_per_h2':    'Distance Acceleration ← NEW',
    'dist_slope_ci_0_5h':     'Distance Slope ← NEW',
    'eta_hours':              'ETA to Contact (engineered)',
}

X_train = train[features].fillna(0)
y_time, y_event = train['time_to_hit_hours'], train['event']
y_xgb = y_time * np.where(y_event == 1, 1, -1)

# Train single model for SHAP
m = xgb.XGBRegressor(objective='survival:cox', n_estimators=45,
                     max_depth=3, learning_rate=0.05, random_state=42)
m.fit(X_train, y_xgb)

# SHAP values
explainer = shap.TreeExplainer(m)
shap_values = explainer.shap_values(X_train)

# Rename columns for display
X_display = X_train.rename(columns=feature_labels)
shap_display = shap_values  # same shape

# --- Plot 1: Bar Summary (Mean |SHAP|) ---
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#0f0f1a')
ax.set_facecolor('#0f0f1a')

mean_shap = np.abs(shap_values).mean(axis=0)
order = np.argsort(mean_shap)
colors = ['#00d4ff' if '← NEW' in list(feature_labels.values())[i] else '#7c3aed'
          for i in order]

ax.barh(range(len(order)), mean_shap[order], color=colors)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([list(feature_labels.values())[i] for i in order],
                   color='white', fontsize=11)
ax.set_xlabel('Mean |SHAP Value|', color='white', fontsize=12)
ax.set_title('V58 Feature Importance (SHAP)\nPurple = V1 Original | Cyan = Newly Discovered',
             color='white', fontsize=13, pad=15)
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('#333355')
ax.xaxis.label.set_color('white')
ax.grid(axis='x', color='#333355', alpha=0.5)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#7c3aed', label='V1 Original Features'),
                   Patch(facecolor='#00d4ff', label='Newly Discovered Features')]
ax.legend(handles=legend_elements, loc='lower right',
          facecolor='#1a1a2e', labelcolor='white', fontsize=10)

plt.tight_layout()
plt.savefig('shap_bar_v58.png', dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
print("Saved: shap_bar_v58.png")

# Print numerical table too
print("\n=== SHAP IMPORTANCE TABLE ===")
shap_df = pd.DataFrame({
    'Feature': [list(feature_labels.values())[i] for i in np.argsort(mean_shap)[::-1]],
    'Mean |SHAP|': sorted(mean_shap, reverse=True)
})
print(shap_df.to_string(index=False))
