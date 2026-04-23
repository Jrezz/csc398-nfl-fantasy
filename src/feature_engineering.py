"""
Feature engineering: transforms raw weekly stats into model-ready features.
All rolling/expanding features use .shift(1) to prevent data leakage.
"""

import pandas as pd
import numpy as np

POSITIONS = ["QB", "RB", "WR", "TE"]

# Feature sets per position (common + position-specific)
COMMON_FEATURES = [
    "rolling_avg_pts_1",
    "rolling_avg_pts_3",
    "rolling_avg_pts_5",
    "season_avg_pts",
    "opp_def_fp_roll4",
    "implied_team_total",
    "vegas_total",
    "home",
    "snap_pct",
    "injury_encoded",
]

POSITION_FEATURES = {
    "QB": COMMON_FEATURES + [
        "passing_yards_roll3",
        "completion_pct_roll3",
        "ypa_roll3",
        "comp_pct_season_avg",
        "attempts_season_avg",
        "td_rate",
        "int_rate",
        "scramble_threat",
    ],
    "RB": COMMON_FEATURES + [
        "rushing_yards_roll3",
        "carries_roll3",
        "rec_yards_roll3",
        "receptions_roll3",
        "td_rate",
    ],
    "WR": COMMON_FEATURES + [
        "receptions_roll3",
        "rec_yards_roll3",
        "targets_roll3",
        "target_share_roll3",
        "td_rate",
    ],
    "TE": COMMON_FEATURES + [
        "receptions_roll3",
        "rec_yards_roll3",
        "targets_roll3",
        "target_share_roll3",
        "td_rate",
    ],
}

TARGET = "fantasy_points_ppr"


def _shift_roll(series: pd.Series, window: int) -> pd.Series:
    return series.shift(1).rolling(window, min_periods=1).mean()


def _shift_expand(series: pd.Series) -> pd.Series:
    return series.shift(1).expanding(min_periods=1).mean()


def add_opponent_info(df: pd.DataFrame, team_sched: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(
        team_sched,
        left_on=["season", "week", "recent_team"],
        right_on=["season", "week", "team"],
        how="left",
    ).drop(columns=["team"], errors="ignore")
    return df


def add_rolling_fantasy_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player_id", "season", "week"]).copy()
    grp = df.groupby("player_id")[TARGET]
    df["rolling_avg_pts_1"] = grp.transform(lambda x: _shift_roll(x, 1))
    df["rolling_avg_pts_3"] = grp.transform(lambda x: _shift_roll(x, 3))
    df["rolling_avg_pts_5"] = grp.transform(lambda x: _shift_roll(x, 5))
    return df


def add_season_features(df: pd.DataFrame) -> pd.DataFrame:
    grp_season = df.groupby(["player_id", "season"])

    df["season_avg_pts"] = grp_season[TARGET].transform(_shift_expand)

    # QB-specific season averages
    if "completions" in df.columns and "attempts" in df.columns:
        df["_season_comp"] = grp_season["completions"].transform(_shift_expand)
        df["_season_att"] = grp_season["attempts"].transform(_shift_expand)
        df["comp_pct_season_avg"] = (df["_season_comp"] / df["_season_att"].replace(0, np.nan)).fillna(0)
        df["attempts_season_avg"] = df["_season_att"]
        df.drop(columns=["_season_comp", "_season_att"], inplace=True)

    # TD rate (TDs per touch — passing_tds + rushing_tds + receiving_tds / attempts + carries + targets)
    td_cols = [c for c in ["passing_tds", "rushing_tds", "receiving_tds"] if c in df.columns]
    usage_cols = [c for c in ["attempts", "carries", "targets"] if c in df.columns]
    if td_cols and usage_cols:
        df["_tds"] = df[td_cols].sum(axis=1)
        df["_usage"] = df[usage_cols].sum(axis=1)
        df["_tds_season"] = grp_season["_tds"].transform(_shift_expand)
        df["_usage_season"] = grp_season["_usage"].transform(_shift_expand)
        df["td_rate"] = (df["_tds_season"] / df["_usage_season"].replace(0, np.nan)).fillna(0)
        df.drop(columns=["_tds", "_usage", "_tds_season", "_usage_season"], inplace=True)
    else:
        df["td_rate"] = 0.0

    # INT rate for QBs
    if "interceptions" in df.columns and "attempts" in df.columns:
        df["_int_season"] = grp_season["interceptions"].transform(_shift_expand)
        df["_att_season"] = grp_season["attempts"].transform(_shift_expand)
        df["int_rate"] = (df["_int_season"] / df["_att_season"].replace(0, np.nan)).fillna(0)
        df.drop(columns=["_int_season", "_att_season"], inplace=True)
    else:
        df["int_rate"] = 0.0

    return df


def add_rolling_position_features(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("player_id")

    # QB rolling
    if "passing_yards" in df.columns:
        df["passing_yards_roll3"] = grp["passing_yards"].transform(lambda x: _shift_roll(x, 3))
    if "completions" in df.columns and "attempts" in df.columns:
        df["_comp_r3"] = grp["completions"].transform(lambda x: _shift_roll(x, 3))
        df["_att_r3"] = grp["attempts"].transform(lambda x: _shift_roll(x, 3))
        df["completion_pct_roll3"] = (df["_comp_r3"] / df["_att_r3"].replace(0, np.nan)).fillna(0)
        df["ypa_roll3"] = (
            grp["passing_yards"].transform(lambda x: _shift_roll(x, 3))
            / df["_att_r3"].replace(0, np.nan)
        ).fillna(0)
        df.drop(columns=["_comp_r3", "_att_r3"], inplace=True)

    # QB scramble threat: rolling rush yards
    if "rushing_yards" in df.columns:
        df["scramble_threat"] = grp["rushing_yards"].transform(lambda x: _shift_roll(x, 3))
        df["rushing_yards_roll3"] = df["scramble_threat"]

    # RB/WR/TE rolling
    if "carries" in df.columns:
        df["carries_roll3"] = grp["carries"].transform(lambda x: _shift_roll(x, 3))
    if "receiving_yards" in df.columns:
        df["rec_yards_roll3"] = grp["receiving_yards"].transform(lambda x: _shift_roll(x, 3))
    if "receptions" in df.columns:
        df["receptions_roll3"] = grp["receptions"].transform(lambda x: _shift_roll(x, 3))
    if "targets" in df.columns:
        df["targets_roll3"] = grp["targets"].transform(lambda x: _shift_roll(x, 3))
    if "target_share" in df.columns:
        df["target_share_roll3"] = grp["target_share"].transform(lambda x: _shift_roll(x, 3))

    return df


def add_defense_features(df: pd.DataFrame) -> pd.DataFrame:
    """4-game rolling avg of fantasy points each defense allowed, by position."""
    if "opponent_team" not in df.columns:
        df["opp_def_fp_roll4"] = np.nan
        return df

    allowed = (
        df.groupby(["season", "week", "opponent_team", "position"])[TARGET]
        .sum()
        .reset_index()
        .rename(columns={"opponent_team": "team", TARGET: "fp_allowed"})
    )
    allowed = allowed.sort_values(["team", "position", "season", "week"])
    allowed["opp_def_fp_roll4"] = allowed.groupby(["team", "position"])["fp_allowed"].transform(
        lambda x: _shift_roll(x, 4)
    )

    df = df.merge(
        allowed[["season", "week", "team", "position", "opp_def_fp_roll4"]],
        left_on=["season", "week", "opponent_team", "position"],
        right_on=["season", "week", "team", "position"],
        how="left",
    ).drop(columns=["team"], errors="ignore")

    return df


def add_injury_features(df: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    STATUS_MAP = {
        "Out": 0,
        "Doubtful": 1,
        "Questionable": 2,
        "Limited": 3,
        "Full Participation in Practice": 4,
        "Full": 4,
    }

    if injuries is None or injuries.empty:
        df["injury_encoded"] = 4
        return df

    id_col = next((c for c in ["gsis_id", "player_id"] if c in injuries.columns), None)
    if id_col is None:
        df["injury_encoded"] = 4
        return df

    inj = injuries[[id_col, "season", "week", "report_status"]].copy()
    inj["injury_encoded"] = inj["report_status"].map(STATUS_MAP).fillna(4)
    inj = inj.groupby([id_col, "season", "week"])["injury_encoded"].min().reset_index()

    df = df.merge(
        inj.rename(columns={id_col: "player_id"}),
        on=["player_id", "season", "week"],
        how="left",
    )
    if "injury_encoded" not in df.columns:
        df["injury_encoded"] = 4
    else:
        df["injury_encoded"] = df["injury_encoded"].fillna(4)

    return df


def fill_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = set()
    for feats in POSITION_FEATURES.values():
        feature_cols.update(feats)
    feature_cols.add(TARGET)

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(df[col].median() if df[col].dtype != object else 0)

    return df


def engineer_features(
    weekly: pd.DataFrame,
    team_sched: pd.DataFrame,
    injuries: pd.DataFrame,
) -> pd.DataFrame:
    print("  Adding opponent/Vegas info...")
    df = add_opponent_info(weekly, team_sched)

    print("  Computing rolling fantasy point averages...")
    df = add_rolling_fantasy_features(df)

    print("  Computing season-to-date features...")
    df = add_season_features(df)

    print("  Computing position-specific rolling features...")
    df = add_rolling_position_features(df)

    print("  Computing opponent defensive ratings...")
    df = add_defense_features(df)

    print("  Adding injury features...")
    df = add_injury_features(df, injuries)

    # snap_pct passthrough
    if "snap_pct" not in df.columns:
        df["snap_pct"] = 0.0
    else:
        df["snap_pct"] = df["snap_pct"].fillna(0.0)

    print("  Filling missing values...")
    df = fill_missing_features(df)

    # Drop rows with no target
    df = df.dropna(subset=[TARGET])

    print(f"  Feature engineering complete: {len(df):,} rows")
    return df
