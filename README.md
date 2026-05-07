# NFL Fantasy Points Prediction

**CSC 398 — Spring 2026 | Saint Cloud State University**  
Justin Rzepko · Tiago Freitas · Jeremiah Trail

Predicts weekly PPR fantasy football scores for NFL skill-position players (QB, RB, WR, TE) using Ridge Regression, K-Nearest Neighbors, Decision Tree, and Random Forest models trained on nflverse data from 2010–2024.

---

## Quickstart — View the Dashboard

Pre-computed results are already committed to the repository (`data/processed/` and `results/`), so you can launch the dashboard immediately without re-running the full pipeline.

**Step 1 — Create a virtual environment**

```bash
python3.11 -m venv .venv
```

If Python 3.11 is not your system default, point to the correct binary (e.g. `python3.11` on macOS with Homebrew). The `runtime.txt` file records the expected version.

**Step 2 — Install dependencies**

```bash
.venv/bin/pip install -r requirements.txt
```

**Step 3 — Launch the dashboard**

```bash
.venv/bin/streamlit run dashboard/app.py
```

The dashboard opens at `http://localhost:8501` in your browser.

---

## Re-running the Full Pipeline

The pipeline downloads raw data from nflverse, engineers features, trains all four models, and overwrites `data/processed/` and `results/`. This step requires `nfl_data_py` and an internet connection. It takes roughly 5–10 minutes on the first run (data is cached afterward).

**Install the extra data dependency**

```bash
.venv/bin/pip install nfl_data_py>=4.3
```

**Run the pipeline**

```bash
.venv/bin/python run_pipeline.py
```

Add `--force` to re-download raw data even if the local cache already exists:

```bash
.venv/bin/python run_pipeline.py --force
```

Then launch the dashboard as shown above.

---

## Project Structure

```
csc398_finalproj/
├── src/
│   ├── data_pipeline.py        # Downloads and caches weekly stats, schedules, injuries
│   ├── feature_engineering.py  # Builds rolling averages, Vegas lines, opponent ratings, etc.
│   └── models.py               # Trains and evaluates all four models; saves results
├── dashboard/
│   └── app.py                  # Streamlit dashboard (5 tabs)
├── data/
│   ├── raw/                    # Cached nflverse downloads (gitignored; re-created by pipeline)
│   └── processed/
│       └── features.csv        # Engineered feature set — 77,768 player-week rows
├── results/
│   ├── metrics.json            # MAE, RMSE, R² for each model × position combination
│   ├── feature_importance.json # Random Forest feature importances by position
│   └── predictions.csv         # Per-player-week predictions on the 2024 holdout set
├── run_pipeline.py             # Entry point: runs data → features → models in sequence
├── launch_dashboard.sh         # Shell shortcut: runs pipeline if needed, then opens dashboard
├── requirements.txt            # Core dependencies (nfl_data_py listed separately above)
└── runtime.txt                 # Python version pin (3.11)
```

---

## Data Sources

All data is fetched from **nflverse** via the `nfl_data_py` package.

| Table | Content | Used for |
|---|---|---|
| `import_weekly_data()` | Per-player per-game stats, 80+ columns | Target variable (PPR points) and base features |
| `import_schedules()` | Game-level spread and over/under lines | Vegas implied team totals, home/away indicator |
| `import_injuries()` | Weekly injury practice status | `injury_encoded` feature (0 = Out, 4 = Full) |

Training set: **2010–2023**  
Holdout test set: **2024** (withheld during all model fitting and cross-validation)

---

## Models

Each position (QB, RB, WR, TE) gets its own set of four trained models. All models are wrapped in a `sklearn.Pipeline` that applies `StandardScaler` before fitting, ensuring that feature scaling is learned only from training data.

| Model | Key settings |
|---|---|
| Ridge Regression | `RidgeCV` with alpha in {0.1, 1, 10, 100} |
| K-Nearest Neighbors | k=10, distance-weighted |
| Decision Tree | max_depth=8, min_samples_leaf=20 |
| Random Forest | 200 trees, max_depth=10, min_samples_leaf=10 |

Cross-validation uses `TimeSeriesSplit` (5-fold) so that each validation fold always comes after its training fold in time, matching real deployment conditions.

---

## Results Summary

Best holdout MAE by position (2024 season):

| Position | Best Model | MAE | R² |
|---|---|---|---|
| QB | Random Forest | 5.99 pts | 0.336 |
| RB | Random Forest / Ridge | 4.83 pts | 0.383 |
| WR | Random Forest | 4.91 pts | 0.290 |
| TE | Random Forest | 3.85 pts | 0.264 |

Random Forest is the top performer across all four positions.

---

## Dashboard Tabs

| Tab | Contents |
|---|---|
| Model Comparison | MAE / RMSE / R² bar charts; CV vs. holdout comparison |
| Predictions | Actual vs. predicted scatter; residual histogram; all-model grid |
| EDA | Score distributions, season trends, correlation heatmap |
| Feature Importance | Random Forest importance by position |
| Rolling Window | Naive MAE comparison across 1-, 3-, 5-game and season-average windows |

Use the sidebar to filter by position or model. All charts update immediately.
