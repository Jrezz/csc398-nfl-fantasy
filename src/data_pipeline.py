"""
Data pipeline: fetch NFL weekly stats, schedules, and injury data from nflverse.
Results are cached to data/raw/ so subsequent runs are instant.
"""

import os
import pandas as pd
import numpy as np

CACHE_DIR = "data/raw"
YEARS = list(range(2010, 2026))
POSITIONS = ["QB", "RB", "WR", "TE"]


def _cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, f"{name}.csv")


def load_weekly_stats(years: list = YEARS, force: bool = False) -> pd.DataFrame:
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
            print(f"    {yr}: skipped ({e})")
    if not frames:
        raise RuntimeError("No weekly data could be downloaded.")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["position"].isin(POSITIONS)].copy()

    # Ensure PPR column exists; compute if missing
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
    Explode schedule into per-team rows with opponent_team, home flag,
    spread_line, total_line, and implied Vegas team totals.
    """
    cols = ["season", "week", "home_team", "away_team"]
    for c in ["spread_line", "total_line"]:
        if c in schedules.columns:
            cols.append(c)

    sched = schedules[cols].dropna(subset=["home_team", "away_team"]).copy()

    total = sched.get("total_line", pd.Series(dtype=float))
    spread = sched.get("spread_line", pd.Series(dtype=float))  # home perspective (neg = home fav)

    # home implied = (total - spread) / 2   away implied = (total + spread) / 2
    sched["home_implied"] = (sched.get("total_line", np.nan) - sched.get("spread_line", np.nan)) / 2
    sched["away_implied"] = (sched.get("total_line", np.nan) + sched.get("spread_line", np.nan)) / 2

    home_rows = sched[["season", "week", "home_team", "away_team", "total_line", "home_implied"]].copy()
    home_rows.columns = ["season", "week", "team", "opponent_team", "vegas_total", "implied_team_total"]
    home_rows["home"] = 1

    away_rows = sched[["season", "week", "away_team", "home_team", "total_line", "away_implied"]].copy()
    away_rows.columns = ["season", "week", "team", "opponent_team", "vegas_total", "implied_team_total"]
    away_rows["home"] = 0

    return pd.concat([home_rows, away_rows], ignore_index=True)


def load_all_data(force: bool = False):
    """Return (weekly_df, team_schedule_lookup, injuries_df)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    weekly = load_weekly_stats(force=force)
    schedules = load_schedules(force=force)
    injuries = load_injuries(force=force)
    team_sched = build_team_schedule_lookup(schedules)
    return weekly, team_sched, injuries
