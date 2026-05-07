"""
Main pipeline entry point.

Runs three stages in order:
  1. Load raw data from nflverse (or local cache).
  2. Engineer features and write data/processed/features.csv.
  3. Train all models and write results/ (metrics, importances, predictions).

After this completes, launch the dashboard with:
    streamlit run dashboard/app.py
"""

import os
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="NFL Fantasy Prediction Pipeline")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download raw data from nflverse even if a local cache exists.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("NFL Fantasy Performance Prediction Pipeline")
    print("CSC 398 · Spring 2026")
    print("=" * 60)

    # Stage 1: data loading
    # Imports are deferred inside each stage so that missing optional packages
    # (like nfl_data_py) only raise an error when the pipeline actually runs,
    # not when the module is imported by the dashboard.
    print("\n[1/3] Loading data...")
    from src.data_pipeline import load_all_data
    weekly, team_sched, injuries = load_all_data(force=args.force)
    seasons = sorted(weekly["season"].unique())
    print(f"  Weekly stats: {len(weekly):,} rows | Seasons: {seasons[0]}–{seasons[-1]}")

    # Stage 2: feature engineering
    print("\n[2/3] Engineering features...")
    from src.feature_engineering import engineer_features
    df = engineer_features(weekly, team_sched, injuries)

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/features.csv", index=False)
    print(f"  Features saved to data/processed/features.csv ({len(df):,} rows)")

    # Stage 3: model training and evaluation
    print("\n[3/3] Training & evaluating models...")
    from src.models import run_all_models
    metrics = run_all_models(df)

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("Launch dashboard:  streamlit run dashboard/app.py")
    print("=" * 60)

    # Print a quick summary table so results are visible without opening the dashboard
    print("\nModel Performance Summary (Holdout MAE by Position):\n")
    header = f"{'Model':<20}" + "".join(f"{p:>10}" for p in ["QB", "RB", "WR", "TE"])
    print(header)
    print("-" * 60)
    for model_name in ["Ridge", "KNN", "Decision Tree", "Random Forest"]:
        row = f"{model_name:<20}"
        for pos in ["QB", "RB", "WR", "TE"]:
            val = metrics.get(pos, {}).get(model_name, {}).get("mae", None)
            row += f"{val:>10.2f}" if val is not None else f"{'N/A':>10}"
        print(row)


if __name__ == "__main__":
    main()
