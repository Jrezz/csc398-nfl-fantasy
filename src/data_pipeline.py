"""
Data pipeline: fetch NFL weekly stats, schedules, and injury data from nflverse.

Each loader checks for a local CSV cache before hitting the network. Subsequent
runs are essentially instant because nflverse downloads are slow (several seconds
per year). Pass force=True to re-download from scratch.

Typical call sequence (from run_pipeline.py):
    weekly, team_sched, injuries = load_all_data()
"""

import os
import pandas as pd
import numpy as np

CACHE_DIR = "data/raw"
# 2025 data is not yet available on nflverse, so the range stops at 2025
# (exclusive) — the last season loaded will be 2024.
YEARS = list(range(2010, 2026))
POSITIONS = ["QB", "RB", "WR", "TE"]


def _cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, f"{name}.csv")


def load_weekly_stats(years: list = YEARS, force: bool = False) -> pd.DataFrame:
    """
    Load per-player per-game statistics for every season in `years`.

    The raw download contains every NFL position (including kickers, linemen,
    etc.). We immediately filter to the four fantasy-relevant positions so the
    cache stays manageable and downstream code doesn't have to handle irrelevant
    rows.

    PPR points are recalculated from component stats if the column is missing.
    nflverse includes it for recent seasons but it can be absent for older data.
    """
    path = _cache_path("weekly_stats")
    if os.path.exists(path) and not force:
        print(f"  Loading weekly stats from cache: {path}")
        return pd.read_csv(path, low_memory=False)

    print("  Downloading weekly stats from nflverse (year by year)...")
    import nfl_data_py as nfl

    frames = []
    for yr in years:
        try:
            chunk = nfl.import_weekly_data(years=[yr])
            frames.append(chunk)
            print(f"    {yr}: {len(chunk):,} rows")
        except Exception as e:
            # Older seasons occasionally have gaps; skip rather than abort.
            print(f"    {yr}: skipped ({e})")

    if not frames:
        raise RuntimeError("No weekly data could be downloaded.")

    df = pd.concat(frames, ignore_index=True)
    df = df[df["position"].isin(POSITIONS)].copy()

    # Recompute PPR if the column is absent. The formula is standard ESPN PPR:
    #   1 pt per reception, 0.1 per rushing/receiving yard, 6 per TD,
    #   0.04 per passing yard, 4 per passing TD, -2 per interception.
    if "fantasy_points_ppr" not in df.columns:
        df["fantasy_points_ppr"] = (
            df.get("receptions", 0) * 1.0
            + df.get("receiving_yards", 0) * 0.1
            + df.get("receiving_tds", 0) * 6.0
            + df.get("rushing_yards", 0) * 0.1
            + df.get("rushing_tds", 0) * 6.0
            + df.get("passing_yards", 0) * 0.04
            + df.get("passing_tds", 0) * 4.0
            + df.get("interceptions", 0) * -2.0
        )

    df.to_csv(path, index=False)
    print(f"  Saved {len(df):,} rows to {path}")
    return df


def load_schedules(years: list = YEARS, force: bool = False) -> pd.DataFrame:
    """
    Load game-level schedule data including Vegas lines.

    We keep only regular-season games (game_type == 'REG') because playoff
    game dynamics differ enough that including them could add noise — and
    fantasy leagues don't run during playoffs anyway.
    """
    path = _cache_path("schedules")
    if os.path.exists(path) and not force:
        print(f"  Loading schedules from cache: {path}")
        return pd.read_csv(path, low_memory=False)

    print("  Downloading schedules from nflverse (year by year)...")
    import nfl_data_py as nfl

    frames = []
    for yr in years:
        try:
            chunk = nfl.import_schedules(years=[yr])
            frames.append(chunk)
        except Exception as e:
            print(f"    {yr} schedules: skipped ({e})")

    sched = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if "game_type" in sched.columns:
        sched = sched[sched["game_type"] == "REG"].copy()

    sched.to_csv(path, index=False)
    print(f"  Saved {len(sched):,} schedule rows to {path}")
    return sched


def load_injuries(years: list = YEARS, force: bool = False) -> pd.DataFrame:
    """
    Load weekly injury practice participation reports.

    The report_status column is later encoded on a 0–4 ordinal scale in
    feature_engineering.add_injury_features. Keeping the raw strings here
    makes it easier to inspect the cache and adjust the mapping if needed.
    """
    path = _cache_path("injuries")
    if os.path.exists(path) and not force:
        print(f"  Loading injuries from cache: {path}")
        return pd.read_csv(path, low_memory=False)

    print("  Downloading injury reports from nflverse (year by year)...")
    import nfl_data_py as nfl

    frames = []
    for yr in years:
        try:
            chunk = nfl.import_injuries(years=[yr])
            frames.append(chunk)
        except Exception as e:
            print(f"    {yr} injuries: skipped ({e})")

    inj = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    inj.to_csv(path, index=False)
    print(f"  Saved {len(inj):,} injury rows to {path}")
    return inj


def build_team_schedule_lookup(schedules: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the game-level schedule table into a per-team per-week lookup.

    The raw schedule has one row per game with home_team and away_team columns.
    We explode it into two rows per game — one for each team — so we can join
    it onto the weekly player stats by (season, week, team). The join attaches
    the opponent, home/away indicator, total line, and each team's implied score.

    Implied team totals are derived from the spread and over/under:
        home_implied = (total - spread) / 2
        away_implied = (total + spread) / 2
    The spread is expressed from the home team's perspective (negative = home
    favorite), so adding it to the total gives the away expectation and
    subtracting gives the home expectation.
    """
    cols = ["season", "week", "home_team", "away_team"]
    for c in ["spread_line", "total_line"]:
        if c in schedules.columns:
            cols.append(c)

    sched = schedules[cols].dropna(subset=["home_team", "away_team"]).copy()

    sched["home_implied"] = (
        sched.get("total_line", np.nan) - sched.get("spread_line", np.nan)
    ) / 2
    sched["away_implied"] = (
        sched.get("total_line", np.nan) + sched.get("spread_line", np.nan)
    ) / 2

    # Split into two frames — one per team per game — then rename to a common schema.
    home_rows = sched[
        ["season", "week", "home_team", "away_team", "total_line", "home_implied"]
    ].copy()
    home_rows.columns = [
        "season", "week", "team", "opponent_team", "vegas_total", "implied_team_total"
    ]
    home_rows["home"] = 1

    away_rows = sched[
        ["season", "week", "away_team", "home_team", "total_line", "away_implied"]
    ].copy()
    away_rows.columns = [
        "season", "week", "team", "opponent_team", "vegas_total", "implied_team_total"
    ]
    away_rows["home"] = 0

    return pd.concat([home_rows, away_rows], ignore_index=True)


def load_all_data(force: bool = False):
    """
    Top-level loader. Returns (weekly_df, team_schedule_lookup, injuries_df).

    The team_schedule_lookup is already exploded into per-team rows so
    feature_engineering can join it directly without any additional reshaping.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    weekly = load_weekly_stats(force=force)
    schedules = load_schedules(force=force)
    injuries = load_injuries(force=force)
    team_sched = build_team_schedule_lookup(schedules)
    return weekly, team_sched, injuries
