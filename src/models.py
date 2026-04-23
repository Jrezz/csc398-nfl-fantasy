"""
Model training and evaluation.
Trains Ridge, KNN, Decision Tree, and Random Forest on each position.
Uses TimeSeriesSplit (5-fold) on 2010-2024 train data.
Holdout test set: 2025 season.
Results saved to results/.
"""

import os
import json
import numpy as np
import pandas as pd

from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.feature_engineering import POSITION_FEATURES, TARGET

RESULTS_DIR = "results"
POSITIONS = ["QB", "RB", "WR", "TE"]

HOLDOUT_YEAR = 2024


def get_model_configs() -> dict:
    return {
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor(n_neighbors=10, weights="distance")),
        ]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("model", DecisionTreeRegressor(max_depth=8, min_samples_leaf=20, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=10,
                n_jobs=-1,
                random_state=42,
            )),
        ]),
    }


def metrics(y_true, y_pred, avg_score: float) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "relative_mae": round(mae / avg_score if avg_score > 0 else 0, 4),
    }


def timeseries_cv(pipeline: Pipeline, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> dict:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes, rmses, r2s = [], [], []
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        pipeline.fit(X_tr, y_tr)
        preds = pipeline.predict(X_val)
        maes.append(mean_absolute_error(y_val, preds))
        rmses.append(np.sqrt(mean_squared_error(y_val, preds)))
        r2s.append(r2_score(y_val, preds))
    return {
        "cv_mae_mean": round(float(np.mean(maes)), 4),
        "cv_mae_std": round(float(np.std(maes)), 4),
        "cv_rmse_mean": round(float(np.mean(rmses)), 4),
        "cv_r2_mean": round(float(np.mean(r2s)), 4),
    }


def get_rf_feature_importance(pipeline: Pipeline, feature_names: list) -> dict:
    try:
        model = pipeline.named_steps["model"]
        importances = model.feature_importances_
        return dict(sorted(zip(feature_names, importances), key=lambda x: -x[1]))
    except AttributeError:
        return {}


def run_all_models(df: pd.DataFrame) -> dict:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_metrics = {}
    all_importances = {}
    pred_frames = []

    train_df = df[df["season"] < HOLDOUT_YEAR].copy()
    test_df = df[df["season"] == HOLDOUT_YEAR].copy()

    print(f"\n  Train rows: {len(train_df):,} | Test (2025) rows: {len(test_df):,}")

    for pos in POSITIONS:
        print(f"\n--- Position: {pos} ---")
        features = [f for f in POSITION_FEATURES[pos] if f in df.columns]
        all_metrics[pos] = {}
        all_importances[pos] = {}

        pos_train = train_df[train_df["position"] == pos].sort_values(["season", "week"]).copy()
        pos_test = test_df[test_df["position"] == pos].copy()

        if len(pos_train) < 100:
            print(f"  Skipping {pos}: insufficient training data ({len(pos_train)} rows)")
            continue

        X_train = pos_train[features].fillna(0).values
        y_train = pos_train[TARGET].values
        avg_score = float(y_train.mean())

        X_test = pos_test[features].fillna(0).values if len(pos_test) > 0 else None
        y_test = pos_test[TARGET].values if len(pos_test) > 0 else None

        for model_name, pipeline in get_model_configs().items():
            print(f"  Training {model_name}...", end=" ")

            cv_stats = timeseries_cv(pipeline, X_train, y_train)
            pipeline.fit(X_train, y_train)

            holdout_stats = {}
            if X_test is not None and len(X_test) > 0:
                preds = pipeline.predict(X_test)
                holdout_stats = metrics(y_test, preds, avg_score)

                pred_rows = pos_test[["player_id", "player_name", "season", "week", "position"]].copy()
                pred_rows["model"] = model_name
                pred_rows["actual"] = y_test
                pred_rows["predicted"] = preds
                pred_frames.append(pred_rows)

            all_metrics[pos][model_name] = {**cv_stats, **holdout_stats}

            if model_name == "Random Forest":
                all_importances[pos] = get_rf_feature_importance(pipeline, features)

            print(f"CV MAE={cv_stats['cv_mae_mean']:.2f}  Holdout MAE={holdout_stats.get('mae', 'N/A')}")

    # Save metrics
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  Metrics saved to {RESULTS_DIR}/metrics.json")

    # Save feature importances
    with open(os.path.join(RESULTS_DIR, "feature_importance.json"), "w") as f:
        json.dump(all_importances, f, indent=2)

    # Save predictions
    if pred_frames:
        preds_df = pd.concat(pred_frames, ignore_index=True)
        preds_df.to_csv(os.path.join(RESULTS_DIR, "predictions.csv"), index=False)
        print(f"  Predictions saved ({len(preds_df):,} rows)")

    return all_metrics
