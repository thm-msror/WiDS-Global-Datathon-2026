import pandas as pd
import numpy as np

print("Executing V69 'THE FIREWALL SNIPER' (V1 Champion + Physical Post-Processing)...")

# --- 1. Load the Champion (V1) and Test Data ---
v1 = pd.read_csv("submissions/final/submission_calibrated_survival_ensemble_v1.csv")
test = pd.read_csv("dataset/test.csv")

# --- 2. Apply the 5.2km Firewall Rule ---
# Training data shows that every fire closer than 5215m eventually hit.
FIREWALL_THRESHOLD = 5215.0

# Merge V1 with test features to identify the target fires
v69 = v1.merge(test[['event_id', 'dist_min_ci_0_5h']], on='event_id')

# Count how many fires we are "fixing"
fires_to_fix = v69[v69['dist_min_ci_0_5h'] < FIREWALL_THRESHOLD]
print(f"Applying Physical Firewall to {len(fires_to_fix)} fires...")

# Apply the Rule: Force high-probability for guaranteed hits
prob_cols = ['prob_12h', 'prob_24h', 'prob_48h', 'prob_72h']
for col in prob_cols:
    # We use 0.999 to avoid being too aggressive but signaling high confidence
    v69.loc[v69['dist_min_ci_0_5h'] < FIREWALL_THRESHOLD, col] = 0.999

# --- 3. Clean and Save ---
sub_path = "submissions/final/submission_firewall_sniper_v69.csv"
v69_final = v69[['event_id'] + prob_cols]
v69_final.to_csv(sub_path, index=False)

print(f"V69 FIREWALL SNIPER saved to {sub_path}")
print("This model bridges the gap to 0.999 using the 5.2km physical hit threshold.")
