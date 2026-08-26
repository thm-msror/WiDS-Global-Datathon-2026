import pandas as pd
import os
from datetime import datetime

LEADERBOARD_PATH = "docs/model_leaderboard.csv"

def log_run(model_name, metrics):
    """
    Appends a new model run to the local leaderboard.
    metrics: dict with keys matching the leaderboard columns.
    """
    # Define columns
    columns = [
        'Timestamp', 'Model Name', 'C-Index', 
        'Brier@24h', 'Brier@48h', 'Brier@72h', 
        'Weighted Brier', 'Hybrid Score'
    ]
    
    # Ensure docs directory exists
    os.makedirs("docs", exist_ok=True)
    
    # Prepare new row
    new_row = {
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Model Name': model_name,
        **metrics
    }
    new_df = pd.DataFrame([new_row])
    
    # Append or create
    if os.path.exists(LEADERBOARD_PATH):
        leaderboard = pd.read_csv(LEADERBOARD_PATH)
        leaderboard = pd.concat([leaderboard, new_df], ignore_index=True)
    else:
        leaderboard = new_df
        
    # Save
    leaderboard.to_csv(LEADERBOARD_PATH, index=False)
    
    # Also save as Markdown for easy viewing
    md_path = LEADERBOARD_PATH.replace(".csv", ".md")
    with open(md_path, "w") as f:
        f.write("# Local Model Leaderboard\n\n")
        f.write(leaderboard.to_markdown(index=False))
    
    print(f"Run logged to {LEADERBOARD_PATH} and {md_path}")

if __name__ == "__main__":
    # Initialize with historical data if it doesn't exist
    if not os.path.exists(LEADERBOARD_PATH):
        print("Initializing leaderboard with historical baselines...")
        baselines = [
            ("Baseline CPH", 0.8805, 0.0503, 0.1587, 0.1408, 0.1208, 0.8796),
            ("Baseline RSF", 0.8861, 0.0552, 0.1610, 0.1575, 0.1282, 0.8761),
            ("Tuned XGBoost", 0.9413, 0.0152, 0.0341, 0.0267, 0.0262, 0.9655),
            ("Ensemble Blend", 0.9402, 0.0337, 0.0265, 0.0041, 0.0220, 0.9667),
        ]
        for name, ci, b24, b48, b72, wb, hs in baselines:
            log_run(name, {
                'C-Index': ci, 'Brier@24h': b24, 'Brier@48h': b48, 
                'Brier@72h': b72, 'Weighted Brier': wb, 'Hybrid Score': hs
            })
