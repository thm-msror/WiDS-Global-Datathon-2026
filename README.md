# WiDS Datathon 2026 - Wildfire Time-to-Threat Survival Modeling

Predicting wildfire time-to-threat probabilities at 12h, 24h, 48h, and 72h horizons using survival analysis and gradient boosting ensembles. Built for the [WiDS Worldwide Global Datathon 2026](https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26/leaderboard) on Kaggle.

## Problem Statement

Given wildfire perimeter observations (growth rate, closing speed, distance, alignment), predict the probability that a fire reaches a community within each time horizon. Evaluation uses a hybrid metric combining Concordance Index and weighted Brier score.

## Approach

- **XGBoost (survival:cox)** as the primary model with Breslow baseline hazard estimation
- **Multi-seed bagging**: 50 seeds x 5 folds (250 models) for robust probability estimates
- **Isotonic calibration** on out-of-fold predictions per horizon
- **Specialist sub-models** trained only on fire-hit events to capture arrival-time dynamics
- **Feature engineering**: growth momentum, threat alignment, distance transforms, firewall splits
- **Interpretability**: SHAP summary/dependence plots and LIME explanations
- **Baselines explored**: Cox PH (lifelines), Random Survival Forest, CatBoost, stacking ensembles
- **Hyperparameter tuning** via Optuna

## Results

| Leaderboard | Rank | Score   |
| ----------- | ---- | ------- |
| Public      | #674 | 0.96579 |
| Private     | #697 | 0.94916 |

Best local CV: C-Index 0.9413 | Hybrid Score 0.967

Best submission script: `core/ultimate_honest_v78.py` (50-seed bagged XGBoost with isotonic calibration). Full model evolution tracked in `docs/model_leaderboard.md`.

## Project Structure

```
core/             9 milestone model scripts + 3 utilities
notebooks/        5 staged Jupyter notebooks (EDA → modeling → interpretability)
dataset/          Competition CSVs
plots/            SHAP, LIME, and calibration visualizations
docs/             Model leaderboard with local CV scores
log_run.py        Experiment logging utility
requirements.txt  Python dependencies
```

### Model Evolution (core/)

| Script                        | Approach                     | Key Change                          |
| ----------------------------- | ---------------------------- | ----------------------------------- |
| `pure_physics_v5`             | Physics-constrained survival | Baseline with domain features       |
| `calibrated_physics_v10`      | + Isotonic calibration       | Improved probability estimates      |
| `absolute_ensemble_v14`       | Multi-model ensemble         | Combined XGB + RSF + CPH            |
| `complete_v59`                | Optuna + SHAP pipeline       | Automated tuning, feature selection |
| `restoration_masterpiece_v62` | Refined ensemble             | First strong Kaggle submission      |
| `firewall_sniper_v69`         | Specialist sub-models        | Per-horizon expert models           |
| `specialist_stack_v77`        | Stacked specialists          | Meta-learner over specialists       |
| `ultimate_honest_v78`         | **Best submission**          | 50-seed bagged XGB, isotonic cal    |
| `v81_honest_50seed`           | Final iteration              | Production-grade 50-seed version    |

## Quick Start

```bash
pip install -r requirements.txt

# Run the structured notebooks in order
jupyter notebook notebooks/

# Or run a standalone model script
python core/ultimate_honest_v78.py
```

## Tech Stack

Python | XGBoost | scikit-survival | lifelines | scikit-learn | Optuna | SHAP | LIME | pandas | NumPy | matplotlib | seaborn
