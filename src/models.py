"""
Model training and evaluation.

Trains four regression models (Ridge, KNN, Decision Tree, Random Forest) for
each of the four skill positions (QB, RB, WR, TE). Each model is wrapped in a
sklearn Pipeline that includes a StandardScaler so that scaling is always fit
on the training fold and never on the validation or holdout data.

Cross-validation uses TimeSeriesSplit (5-fold) on 2010–2023 training data.
Final evaluation uses the 2024 season as a held-out test set that the models
never see during fitting or cross-validation.

Output files written to results/:
    metrics.json            - MAE, RMSE, R² for each model × position
    feature_importance.json - Random Forest importances (not available for KNN/linear)
    predictions.csv         - Per-player-week holdout predictions for all models
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

# 2024 is withheld as the holdout test year. The choice of 2024 (rather than a
# random split) preserves temporal order: the model is always predicting a future
# season it has never seen, which mirrors real-world deployment.
HOLDOUT_YEAR = 2024


def get_model_configs() -> dict:
    """
    Build and return a fresh set of model Pipelines.

    Each Pipeline wraps the model with StandardScaler. Placing the scaler
    inside the Pipeline means sklearn will always fit it on the training data
    of each fold — if we scaled the whole dataset up front, the scaler would
    'see' the validation data, which is a form of data leakage.

    Hyperparameters were chosen to balance bias and variance without extensive
    tuning, keeping the focus on comparing model families rather than squeezing
    out marginal gains from grid search.
    """
    return {
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            # RidgeCV automatically selects the best alpha via efficient LOOCV,
            # so we don't need a separate grid-search step for this model.
            ("model", RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            # Distance weighting gives closer neighbors more influence, which
            # works better than uniform weighting for continuous outputs.
            ("model", KNeighborsRegressor(n_neighbors=10, weights="distance")),
        ]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            # max_depth=8 and min_samples_leaf=20 are regularization constraints
            # that prevent the tree from memorizing individual training rows.
            ("model", DecisionTreeRegressor(
                max_depth=8, min_samples_leaf=20, random_state=42
            )),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            # 200 trees is enough to stabilize the ensemble's variance without
            # making training prohibitively slow. n_jobs=-1 uses all available
            # CPU cores since each tree is independent.
            ("model", RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=10,
                n_jobs=-1,
                random_state=42,
            )),
        ]),
    }


def compute_metrics(y_true, y_pred, avg_score: float) -> dict:
    """
    Return MAE, RMSE, R², and relative MAE (MAE as a fraction of the mean score).

    Relative MAE contextualizes the absolute error: an MAE of 4 pts is a lot
    when the average score is 7 pts (57%) but much less significant when the
    average is 20 pts (20%).
    """
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
    """
    Run k-fold cross-validation while respecting chronological order.

    TimeSeriesSplit ensures that the validation fold always comes after the
    training fold in time. Standard KFold would randomly mix future and past
    data into both folds, causing the model to appear much better in CV than
    it actually would be in production. The growing-window approach means each
    successive fold has more training data but the same validation period length.
    """
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
    """
    Extract feature importances from the Random Forest step of a fitted Pipeline.

    Returns a dict sorted by descending importance so it's ready to serialize.
    Returns an empty dict for models that don't expose feature_importances_
    (Ridge, KNN), rather than raising an exception.
    """
    try:
        model = pipeline.named_steps["model"]
        importances = model.feature_importances_
        return dict(sorted(zip(feature_names, importances), key=lambda x: -x[1]))
    except AttributeError:
        return {}


def run_all_models(df: pd.DataFrame) -> dict:
    """
    Train and evaluate all model × position combinations. Saves three output
    files to results/ and returns the metrics dict for the pipeline summary.

    For each position:
      1. Filter to that position's rows and its specific feature columns.
      2. Run TimeSeriesSplit CV on the training set (2010–2023).
      3. Refit on the full training set, then evaluate on the 2024 holdout.
      4. Collect holdout predictions and Random Forest importances.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_metrics = {}
    all_importances = {}
    pred_frames = []

    train_df = df[df["season"] < HOLDOUT_YEAR].copy()
    test_df = df[df["season"] == HOLDOUT_YEAR].copy()

    print(f"\n  Train rows: {len(train_df):,} | Test ({HOLDOUT_YEAR}) rows: {len(test_df):,}")

    for pos in POSITIONS:
        print(f"\n--- Position: {pos} ---")

        # Only use features that actually exist in the dataframe. Some columns
        # may be absent if the nflverse download is incomplete for older seasons.
        features = [f for f in POSITION_FEATURES[pos] if f in df.columns]
        all_metrics[pos] = {}
        all_importances[pos] = {}

        # Sort by season/week so TimeSeriesSplit sees data in chronological order.
        pos_train = (
            train_df[train_df["position"] == pos]
            .sort_values(["season", "week"])
            .copy()
        )
        pos_test = test_df[test_df["position"] == pos].copy()

        if len(pos_train) < 100:
            print(f"  Skipping {pos}: insufficient training data ({len(pos_train)} rows)")
            continue

        X_train = pos_train[features].fillna(0).values
        y_train = pos_train[TARGET].values
        avg_score = float(y_train.mean())

        X_test = pos_test[features].fillna(0).values if len(pos_test) > 0 else None
        y_test = pos_test[TARGET].values if len(pos_test) > 0 else None

        # get_model_configs() is called inside the loop to ensure each position
        # starts with fresh, unfitted pipeline objects.
        for model_name, pipeline in get_model_configs().items():
            print(f"  Training {model_name}...", end=" ")

            cv_stats = timeseries_cv(pipeline, X_train, y_train)

            # Refit on the full training set after CV so the final model has
            # seen as much data as possible before evaluating on the holdout.
            pipeline.fit(X_train, y_train)

            holdout_stats = {}
            if X_test is not None and len(X_test) > 0:
                preds = pipeline.predict(X_test)
                holdout_stats = compute_metrics(y_test, preds, avg_score)

                # Retain the identifying columns alongside each prediction so the
                # dashboard can display player names, filter by week, etc.
                pred_rows = pos_test[
                    ["player_id", "player_name", "season", "week", "position"]
                ].copy()
                pred_rows["model"] = model_name
                pred_rows["actual"] = y_test
                pred_rows["predicted"] = preds
                pred_frames.append(pred_rows)

            all_metrics[pos][model_name] = {**cv_stats, **holdout_stats}

            # Feature importances are only meaningful for tree-based models;
            # the helper returns {} for Ridge and KNN.
            if model_name == "Random Forest":
                all_importances[pos] = get_rf_feature_importance(pipeline, features)

            print(
                f"CV MAE={cv_stats['cv_mae_mean']:.2f}  "
                f"Holdout MAE={holdout_stats.get('mae', 'N/A')}"
            )

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  Metrics saved to {RESULTS_DIR}/metrics.json")

    with open(os.path.join(RESULTS_DIR, "feature_importance.json"), "w") as f:
        json.dump(all_importances, f, indent=2)

    if pred_frames:
        preds_df = pd.concat(pred_frames, ignore_index=True)
        preds_df.to_csv(os.path.join(RESULTS_DIR, "predictions.csv"), index=False)
        print(f"  Predictions saved ({len(preds_df):,} rows)")

    return all_metrics
