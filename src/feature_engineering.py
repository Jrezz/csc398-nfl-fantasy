"""
Feature engineering: transforms raw weekly stats into model-ready features.

The central design constraint throughout this file is preventing data leakage.
Every rolling or expanding statistic is shifted by one period before the window
is applied, so the value available to the model for a given game only reflects
what was known before that game was played. See _shift_roll and _shift_expand.

Feature sets are position-specific because the statistics that predict a WR's
week differ substantially from those that predict a QB's week. A shared feature
set would force irrelevant columns to be imputed as zero, adding noise.
"""

import pandas as pd
import numpy as np

POSITIONS = ["QB", "RB", "WR", "TE"]

# Features shared across all positions.
# These capture overall player form, game environment, and health.
COMMON_FEATURES = [
    "rolling_avg_pts_1",     # last game's PPR score (1-game lag)
    "rolling_avg_pts_3",     # average over the prior 3 games
    "rolling_avg_pts_5",     # average over the prior 5 games
    "season_avg_pts",        # expanding season-to-date average
    "opp_def_fp_roll4",      # how many fantasy pts this defense allowed over the last 4 weeks
    "implied_team_total",    # Vegas implied team score (proxy for game script)
    "vegas_total",           # over/under for the game
    "home",                  # 1 if home, 0 if away
    "snap_pct",              # share of offensive snaps (availability / usage)
    "injury_encoded",        # practice status on a 0 (Out) to 4 (Full) ordinal scale
]

POSITION_FEATURES = {
    "QB": COMMON_FEATURES + [
        "passing_yards_roll3",    # recent passing volume
        "completion_pct_roll3",   # recent accuracy
        "ypa_roll3",              # yards per attempt — efficiency signal
        "comp_pct_season_avg",    # season-long completion rate baseline
        "attempts_season_avg",    # season-long workload baseline
        "td_rate",                # TDs per attempt (season expanding)
        "int_rate",               # interceptions per attempt (season expanding)
        "scramble_threat",        # rolling rushing yards — measures QB mobility
    ],
    "RB": COMMON_FEATURES + [
        "rushing_yards_roll3",
        "carries_roll3",
        "rec_yards_roll3",
        "receptions_roll3",
        "td_rate",                # TDs per touch (rushing + receiving)
    ],
    "WR": COMMON_FEATURES + [
        "receptions_roll3",
        "rec_yards_roll3",
        "targets_roll3",
        "target_share_roll3",     # share of team targets — more stable than raw counts
        "td_rate",                # TDs per target
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
    """
    Shift a series by one period then apply a rolling mean.

    The shift ensures the model never sees the current game's outcome when
    computing the feature. Without it, the rolling average for week N would
    include week N's actual score, which is the target we're trying to predict.
    """
    return series.shift(1).rolling(window, min_periods=1).mean()


def _shift_expand(series: pd.Series) -> pd.Series:
    """
    Expanding (cumulative) mean, shifted by one period for the same reason.

    Used for season-to-date averages where we want all prior games in the
    current season but not the current game.
    """
    return series.shift(1).expanding(min_periods=1).mean()


def add_opponent_info(df: pd.DataFrame, team_sched: pd.DataFrame) -> pd.DataFrame:
    """
    Join schedule data onto player stats so each row knows its opponent and
    whether the player was home or away, plus the Vegas lines for that game.
    """
    df = df.merge(
        team_sched,
        left_on=["season", "week", "recent_team"],
        right_on=["season", "week", "team"],
        how="left",
    ).drop(columns=["team"], errors="ignore")
    return df


def add_rolling_fantasy_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling PPR averages across all positions.

    Grouping by player_id (not player_name) handles name changes and players
    who share the same name. Sorting by season then week ensures the shift
    correctly identifies 'the previous game' even across season boundaries.
    """
    df = df.sort_values(["player_id", "season", "week"]).copy()
    grp = df.groupby("player_id")[TARGET]
    df["rolling_avg_pts_1"] = grp.transform(lambda x: _shift_roll(x, 1))
    df["rolling_avg_pts_3"] = grp.transform(lambda x: _shift_roll(x, 3))
    df["rolling_avg_pts_5"] = grp.transform(lambda x: _shift_roll(x, 5))
    return df


def add_season_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Season-to-date expanding averages for PPR score and key per-position rates.

    We group by (player_id, season) rather than just player_id so the expanding
    window resets at the start of each new season. A player's 2019 statistics
    shouldn't influence their 2020 season average — that's what the cross-season
    rolling features are for.
    """
    grp_season = df.groupby(["player_id", "season"])

    df["season_avg_pts"] = grp_season[TARGET].transform(_shift_expand)

    # Completion percentage and attempt volume for QBs.
    # These are computed as expanding averages of the raw counts, then divided,
    # rather than expanding the per-game percentages. This avoids giving equal
    # weight to a 2/2 game (100%) and a 25/30 game (83%).
    if "completions" in df.columns and "attempts" in df.columns:
        df["_season_comp"] = grp_season["completions"].transform(_shift_expand)
        df["_season_att"] = grp_season["attempts"].transform(_shift_expand)
        df["comp_pct_season_avg"] = (
            df["_season_comp"] / df["_season_att"].replace(0, np.nan)
        ).fillna(0)
        df["attempts_season_avg"] = df["_season_att"]
        df.drop(columns=["_season_comp", "_season_att"], inplace=True)

    # TD rate: season-expanding TDs divided by season-expanding usage.
    # For a QB, 'usage' is attempts; for an RB it's carries + targets, etc.
    # We sum whichever columns exist so the same code path works for all positions.
    td_cols = [c for c in ["passing_tds", "rushing_tds", "receiving_tds"] if c in df.columns]
    usage_cols = [c for c in ["attempts", "carries", "targets"] if c in df.columns]
    if td_cols and usage_cols:
        df["_tds"] = df[td_cols].sum(axis=1)
        df["_usage"] = df[usage_cols].sum(axis=1)
        df["_tds_season"] = grp_season["_tds"].transform(_shift_expand)
        df["_usage_season"] = grp_season["_usage"].transform(_shift_expand)
        df["td_rate"] = (
            df["_tds_season"] / df["_usage_season"].replace(0, np.nan)
        ).fillna(0)
        df.drop(columns=["_tds", "_usage", "_tds_season", "_usage_season"], inplace=True)
    else:
        df["td_rate"] = 0.0

    # Interception rate for QBs — separate from td_rate because it's a negative
    # signal and mixing it into td_rate would obscure both signals.
    if "interceptions" in df.columns and "attempts" in df.columns:
        df["_int_season"] = grp_season["interceptions"].transform(_shift_expand)
        df["_att_season"] = grp_season["attempts"].transform(_shift_expand)
        df["int_rate"] = (
            df["_int_season"] / df["_att_season"].replace(0, np.nan)
        ).fillna(0)
        df.drop(columns=["_int_season", "_att_season"], inplace=True)
    else:
        df["int_rate"] = 0.0

    return df


def add_rolling_position_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Three-game rolling averages for position-specific counting stats.

    A 3-game window captures recent usage trends without being as noisy as a
    single-game value, while remaining more responsive than a season average.
    """
    grp = df.groupby("player_id")

    # QB volume and efficiency
    if "passing_yards" in df.columns:
        df["passing_yards_roll3"] = grp["passing_yards"].transform(
            lambda x: _shift_roll(x, 3)
        )
    if "completions" in df.columns and "attempts" in df.columns:
        df["_comp_r3"] = grp["completions"].transform(lambda x: _shift_roll(x, 3))
        df["_att_r3"] = grp["attempts"].transform(lambda x: _shift_roll(x, 3))
        df["completion_pct_roll3"] = (
            df["_comp_r3"] / df["_att_r3"].replace(0, np.nan)
        ).fillna(0)
        # Yards per attempt over the last 3 games — captures efficiency independent
        # of volume (a QB with 300 yards on 50 attempts is different from one with
        # 300 yards on 30 attempts).
        df["ypa_roll3"] = (
            grp["passing_yards"].transform(lambda x: _shift_roll(x, 3))
            / df["_att_r3"].replace(0, np.nan)
        ).fillna(0)
        df.drop(columns=["_comp_r3", "_att_r3"], inplace=True)

    # QB rushing — 'scramble_threat' doubles as rushing_yards_roll3 for RBs.
    # We create both column names pointing to the same values so the QB and RB
    # feature lists can each reference the name appropriate to the position.
    if "rushing_yards" in df.columns:
        df["scramble_threat"] = grp["rushing_yards"].transform(
            lambda x: _shift_roll(x, 3)
        )
        df["rushing_yards_roll3"] = df["scramble_threat"]

    # RB workload
    if "carries" in df.columns:
        df["carries_roll3"] = grp["carries"].transform(lambda x: _shift_roll(x, 3))

    # Receiving stats (used by RB, WR, TE)
    if "receiving_yards" in df.columns:
        df["rec_yards_roll3"] = grp["receiving_yards"].transform(
            lambda x: _shift_roll(x, 3)
        )
    if "receptions" in df.columns:
        df["receptions_roll3"] = grp["receptions"].transform(
            lambda x: _shift_roll(x, 3)
        )
    if "targets" in df.columns:
        df["targets_roll3"] = grp["targets"].transform(lambda x: _shift_roll(x, 3))
    if "target_share" in df.columns:
        # Target share (player targets / team targets) is more stable than raw
        # target counts because it adjusts for pace and game script automatically.
        df["target_share_roll3"] = grp["target_share"].transform(
            lambda x: _shift_roll(x, 3)
        )

    return df


def add_defense_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute how many fantasy points each defense has allowed per position over
    the last 4 weeks, then attach that rating to each player's upcoming game.

    This is a rough measure of matchup quality: a WR facing a defense that has
    allowed 45 PPR points to opposing WRs per game is in a much better spot
    than one facing a defense that has allowed 20.

    The 4-week window balances responsiveness to recent performance against
    the noise in a small per-week sample.
    """
    if "opponent_team" not in df.columns:
        df["opp_def_fp_roll4"] = np.nan
        return df

    # Sum fantasy points allowed by each defense per position per week.
    # We then roll that across the 4 prior weeks (with shift) before merging
    # back, so the feature for week N reflects weeks N-4 through N-1.
    allowed = (
        df.groupby(["season", "week", "opponent_team", "position"])[TARGET]
        .sum()
        .reset_index()
        .rename(columns={"opponent_team": "team", TARGET: "fp_allowed"})
    )
    allowed = allowed.sort_values(["team", "position", "season", "week"])
    allowed["opp_def_fp_roll4"] = allowed.groupby(["team", "position"])[
        "fp_allowed"
    ].transform(lambda x: _shift_roll(x, 4))

    df = df.merge(
        allowed[["season", "week", "team", "position", "opp_def_fp_roll4"]],
        left_on=["season", "week", "opponent_team", "position"],
        right_on=["season", "week", "team", "position"],
        how="left",
    ).drop(columns=["team"], errors="ignore")

    return df


def add_injury_features(df: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    """
    Encode injury practice participation into an ordinal feature.

    The encoding runs from 0 (fully out) to 4 (full participant / healthy):
        Out                          -> 0
        Doubtful                     -> 1
        Questionable                 -> 2
        Limited                      -> 3
        Full Participation           -> 4
        Missing / unknown            -> 4 (assume healthy unless told otherwise)

    When a player appears multiple times in the same week (e.g., Wednesday and
    Friday reports), we take the minimum — the worst status seen — because that
    is more conservative and better captures actual risk of limited play.
    """
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

    # nflverse uses 'gsis_id' in some seasons and 'player_id' in others.
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
    """
    Ensure every feature column exists and contains no NaN values.

    Missing values arise naturally for early-season rows (there are no prior
    games to roll over) and for players who miss weeks (no snap_pct, for
    example). We fill with the column median rather than zero so that imputed
    values are close to the typical value for that feature, which reduces bias
    compared to zero-filling.

    Columns that don't exist at all (e.g., a position-specific column not
    computed for this dataset) are added as 0.0.
    """
    feature_cols = set()
    for feats in POSITION_FEATURES.values():
        feature_cols.update(feats)
    feature_cols.add(TARGET)

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
        else:
            fill_val = df[col].median() if df[col].dtype != object else 0
            df[col] = df[col].fillna(fill_val)

    return df


def engineer_features(
    weekly: pd.DataFrame,
    team_sched: pd.DataFrame,
    injuries: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run all feature engineering steps in the required order and return the
    final feature-complete DataFrame.

    Steps must run in this order because later steps depend on columns that
    earlier steps create (e.g., add_defense_features needs opponent_team from
    add_opponent_info, and fill_missing_features must run last).
    """
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

    # snap_pct passes through unchanged if it exists; otherwise it defaults to 0
    # so the column is always present for the model feature lists.
    if "snap_pct" not in df.columns:
        df["snap_pct"] = 0.0
    else:
        df["snap_pct"] = df["snap_pct"].fillna(0.0)

    print("  Filling missing values...")
    df = fill_missing_features(df)

    # Drop any rows where the target is still missing after filling.
    # This is rare and typically indicates the player was listed on the roster
    # but did not play (e.g., a healthy scratch).
    df = df.dropna(subset=[TARGET])

    print(f"  Feature engineering complete: {len(df):,} rows")
    return df
