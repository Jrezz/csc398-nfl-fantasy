"""
Main pipeline entry point.
Run: python run_pipeline.py
Then: streamlit run dashboard/app.py
"""

import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="NFL Fantasy Prediction Pipeline")
    parser.add_argument("--force", action="store_true", help="Re-download data even if cached")
    args = parser.parse_args()

    print("=" * 60)
    print("NFL Fantasy Performance Prediction Pipeline")
    print("CSC 398 · Spring 2026")
    print("=" * 60)

    print("\n[1/3] Loading data...")
    from src.data_pipeline import load_all_data
    weekly, team_sched, injuries = load_all_data(force=args.force)
    seasons = sorted(weekly['season'].unique())
    print(f"  Weekly stats: {len(weekly):,} rows | Seasons: {seasons[0]}–{seasons[-1]}")

    print("\n[2/3] Engineering features...")
    from src.feature_engineering import engineer_features
    df = engineer_features(weekly, team_sched, injuries)

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/features.csv", index=False)
    print(f"  Features saved to data/processed/features.csv ({len(df):,} rows)")

    print("\n[3/3] Training & evaluating models...")
    from src.models import run_all_models
    metrics = run_all_models(df)

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("Launch dashboard:  streamlit run dashboard/app.py")
    print("=" * 60)

    # Print summary table
    print("\nModel Performance Summary (Holdout MAE by Position):\n")
    header = f"{'Model':<20}" + "".join(f"{'QB':>10}{'RB':>10}{'WR':>10}{'TE':>10}")
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
