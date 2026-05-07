"""
NFL Fantasy Performance Prediction — Streamlit Dashboard
CSC 398 · Spring 2026  |  streamlit run dashboard/app.py

Reads pre-computed output from run_pipeline.py:
    data/processed/features.csv  - engineered feature set (all seasons)
    results/metrics.json         - model performance by position
    results/feature_importance.json
    results/predictions.csv      - holdout predictions for each model

All charts are built with Plotly so they remain interactive in the browser.
"""

import os
import sys
import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Resolve the project root regardless of where streamlit is launched from.
# os.chdir ensures that relative file paths (e.g., "results/metrics.json")
# work the same way whether the dashboard is started from the project root
# or from the dashboard/ subdirectory.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

# set_page_config must be the very first Streamlit call in the script.
# Calling any other st.* function before this raises a StreamlitAPIException.
st.set_page_config(
    page_title="NFL Fantasy · CSC 398",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Color palette ---
# Consistent with GitHub dark theme so charts look good on most monitors.
# These are referenced in both the injected CSS and the Plotly traces.
SURFACE  = "#161b22"
BORDER   = "#30363d"
TEXT_PRI = "#e6edf3"
TEXT_SEC = "#8b949e"
BLUE     = "#58a6ff"
GREEN    = "#3fb950"
GOLD     = "#d29922"
PURPLE   = "#bc8cff"
RED      = "#f85149"

# Each position and model gets a consistent color across all charts so the
# reader doesn't have to re-learn the legend on every tab.
POS_COLOR = {"QB": BLUE, "RB": GREEN, "WR": GOLD, "TE": PURPLE}
MDL_COLOR = {
    "Ridge":         BLUE,
    "KNN":           GREEN,
    "Decision Tree": GOLD,
    "Random Forest": PURPLE,
}

POSITIONS    = ["QB", "RB", "WR", "TE"]
MODELS       = ["Ridge", "KNN", "Decision Tree", "Random Forest"]
HOLDOUT_YEAR = 2024
TARGET       = "fantasy_points_ppr"

# Feature lists are duplicated here (also defined in src/feature_engineering.py)
# so the dashboard has no hard dependency on the pipeline source code. This
# allows the dashboard to run without nfl_data_py installed.
POSITION_FEATURES = {
    "QB": ["rolling_avg_pts_1","rolling_avg_pts_3","rolling_avg_pts_5",
           "season_avg_pts","passing_yards_roll3","completion_pct_roll3",
           "ypa_roll3","comp_pct_season_avg","attempts_season_avg",
           "td_rate","int_rate","scramble_threat","opp_def_fp_roll4",
           "implied_team_total","vegas_total","home","snap_pct","injury_encoded"],
    "RB": ["rolling_avg_pts_1","rolling_avg_pts_3","rolling_avg_pts_5",
           "season_avg_pts","rushing_yards_roll3","carries_roll3",
           "rec_yards_roll3","receptions_roll3","td_rate","opp_def_fp_roll4",
           "implied_team_total","vegas_total","home","snap_pct","injury_encoded"],
    "WR": ["rolling_avg_pts_1","rolling_avg_pts_3","rolling_avg_pts_5",
           "season_avg_pts","receptions_roll3","rec_yards_roll3","targets_roll3",
           "target_share_roll3","td_rate","opp_def_fp_roll4",
           "implied_team_total","vegas_total","home","snap_pct","injury_encoded"],
    "TE": ["rolling_avg_pts_1","rolling_avg_pts_3","rolling_avg_pts_5",
           "season_avg_pts","receptions_roll3","rec_yards_roll3","targets_roll3",
           "target_share_roll3","td_rate","opp_def_fp_roll4",
           "implied_team_total","vegas_total","home","snap_pct","injury_encoded"],
}

# Rolling window labels map to the feature column names computed during
# feature engineering. Used in Tab 5 to compare window sizes as naive baselines.
WINDOWS = {
    "1 Game":     "rolling_avg_pts_1",
    "3 Games":    "rolling_avg_pts_3",
    "5 Games":    "rolling_avg_pts_5",
    "Season Avg": "season_avg_pts",
}

# --- Global CSS ---
# Injected once at startup. Overrides Streamlit's default styling to match
# the dark color palette and tighten spacing on cards, tabs, and tables.
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}}

.block-container {{
    padding: 2rem 2rem 3rem 2rem !important;
    max-width: 1440px !important;
}}

h1, h2, h3 {{ color: {TEXT_PRI} !important; }}
h3 {{ font-size: 1.05rem !important; font-weight: 600 !important; margin-bottom: 0 !important; }}
p {{ color: {TEXT_SEC}; font-size: 0.875rem; }}

section[data-testid="stSidebar"] {{
    background: #0d1117 !important;
    border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] > div {{ padding: 1.5rem 1.25rem; }}

.stTabs [data-baseweb="tab-list"] {{
    gap: 0;
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 4px;
    margin-bottom: 1.5rem;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 6px;
    padding: 7px 18px;
    font-size: 0.825rem;
    font-weight: 500;
    color: {TEXT_SEC};
    border: none !important;
    outline: none !important;
}}
.stTabs [aria-selected="true"] {{
    background: #21262d !important;
    color: {TEXT_PRI} !important;
}}
.stTabs [data-baseweb="tab-highlight"] {{ display: none; }}
.stTabs [data-baseweb="tab-border"]   {{ display: none; }}

.kpi-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 20px 16px 16px;
    text-align: center;
}}
.kpi-num {{
    font-size: 1.9rem;
    font-weight: 700;
    color: {TEXT_PRI};
    line-height: 1;
}}
.kpi-lbl {{
    font-size: 0.68rem;
    color: #6e7681;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-top: 6px;
}}

.overline {{
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {BLUE};
    margin-bottom: 4px;
    display: block;
}}

.insight {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-left: 3px solid {GREEN};
    border-radius: 0 6px 6px 0;
    padding: 10px 14px;
    font-size: 0.83rem;
    color: #c9d1d9;
    margin-bottom: 6px;
}}

.feat-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid {BORDER};
    font-size: 0.8rem;
}}
.feat-name {{ color: {TEXT_PRI}; }}
.feat-val  {{ color: {TEXT_SEC}; font-variant-numeric: tabular-nums; }}

hr {{ border-color: {BORDER} !important; margin: 1.25rem 0 !important; }}

.stDataFrame {{ border-radius: 8px !important; }}
[data-testid="stMetricValue"] {{ font-size: 1.4rem !important; }}
</style>
""", unsafe_allow_html=True)


# --- Shared chart layout ---
# All Plotly figures use this as a base so colors, fonts, and grid lines are
# consistent across tabs without repeating the same kwargs on every figure.
_BASE_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=TEXT_SEC, size=11.5),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11), borderwidth=0),
    hoverlabel=dict(bgcolor="#21262d", font_size=12, bordercolor=BORDER),
    xaxis=dict(gridcolor="#21262d", linecolor=BORDER, tickcolor=BORDER),
    yaxis=dict(gridcolor="#21262d", linecolor=BORDER, tickcolor=BORDER),
)

def chart(**overrides) -> dict:
    """Merge caller overrides into the base layout dict."""
    out = dict(_BASE_LAYOUT)
    out["margin"] = overrides.pop("margin", dict(t=32, b=32, l=16, r=16))
    out["height"]  = overrides.pop("height", 380)
    out.update(overrides)
    return out


# --- Data loaders ---
# @st.cache_data means each file is read from disk only once per session.
# Subsequent interactions that call these functions hit the in-memory cache.
@st.cache_data
def load_metrics():
    p = os.path.join(ROOT, "results", "metrics.json")
    return json.load(open(p)) if os.path.exists(p) else None

@st.cache_data
def load_predictions():
    p = os.path.join(ROOT, "results", "predictions.csv")
    return pd.read_csv(p) if os.path.exists(p) else None

@st.cache_data
def load_importances():
    p = os.path.join(ROOT, "results", "feature_importance.json")
    return json.load(open(p)) if os.path.exists(p) else None

@st.cache_data
def load_features():
    p = os.path.join(ROOT, "data", "processed", "features.csv")
    return pd.read_csv(p, low_memory=False) if os.path.exists(p) else None


metrics_data = load_metrics()
preds_df     = load_predictions()
importances  = load_importances()
features_df  = load_features()


# --- Sidebar ---
with st.sidebar:
    st.markdown("### NFL Fantasy Predictor")
    st.caption("CSC 398 · Spring 2026")
    st.divider()

    sel_positions = st.multiselect("Positions", POSITIONS, default=POSITIONS)
    sel_models    = st.multiselect("Models",    MODELS,    default=MODELS)

    st.divider()
    st.caption("**Team**  \nJustin Rzepko  \nTiago Freitas  \nJeremiah Trail  \n  \nProf. Antonios / Martin")

# Fall back to all options when the user clears a multiselect entirely,
# so charts always have something to display.
active_pos = sel_positions or POSITIONS
active_mdl = sel_models    or MODELS


# --- Page header ---
st.markdown(
    f'<span class="overline">CSC 398 · Spring 2026</span>'
    f'<h1 style="font-size:1.75rem;font-weight:700;margin:4px 0 6px;color:{TEXT_PRI}">'
    f'NFL Fantasy Performance Prediction</h1>'
    f'<p style="color:{TEXT_SEC};margin-bottom:1.5rem">'
    f'Ridge · KNN · Decision Tree · Random Forest &nbsp;|&nbsp; '
    f'QB / RB / WR / TE &nbsp;|&nbsp; 2010–2024 · nflverse</p>',
    unsafe_allow_html=True,
)

if metrics_data is None:
    st.error("Pipeline results not found — run `python run_pipeline.py` first.")
    st.stop()

# KPI row — summary stats at a glance above the tabs
k1, k2, k3, k4 = st.columns(4)
for col, num, lbl in zip(
    [k1, k2, k3, k4],
    ["77,768", "15", "4", "4"],
    ["Player-Week Rows", "NFL Seasons", "Positions Modeled", "ML Models"],
):
    col.markdown(
        f'<div class="kpi-card"><div class="kpi-num">{num}</div>'
        f'<div class="kpi-lbl">{lbl}</div></div>',
        unsafe_allow_html=True,
    )


# --- Tabs ---
st.markdown("<div style='margin-top:1.75rem'></div>", unsafe_allow_html=True)
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Model Comparison", "Predictions", "EDA", "Feature Importance", "Rolling Window"]
)


# ============================================================
# TAB 1 — MODEL COMPARISON
# ============================================================
with tab1:

    # Build a flat DataFrame from the nested metrics dict so Plotly can use it.
    rows = []
    for pos in active_pos:
        for mdl in active_mdl:
            m = metrics_data.get(pos, {}).get(mdl, {})
            if m:
                rows.append(dict(
                    Position=pos, Model=mdl,
                    MAE=m.get("mae"), RMSE=m.get("rmse"),
                    R2=m.get("r2"),  RelMAE=m.get("relative_mae"),
                    CV_MAE=m.get("cv_mae_mean"), CV_Std=m.get("cv_mae_std"),
                ))
    if not rows:
        st.info("No data — select at least one position and model.")
        st.stop()

    mdf = pd.DataFrame(rows)

    # Leaderboard — pivot to model × position and sort by average holdout MAE
    st.markdown('<span class="overline">Overall Rankings</span>', unsafe_allow_html=True)
    st.markdown("### Model Leaderboard — 2024 Holdout MAE")
    st.caption("Ranked by average MAE across all positions (lower = better).")

    pivot = mdf.dropna(subset=["MAE"]).pivot_table(
        index="Model", columns="Position", values="MAE", aggfunc="mean"
    )
    pos_cols = [p for p in POSITIONS if p in pivot.columns]
    pivot = pivot[pos_cols]
    pivot["Avg MAE"] = pivot[pos_cols].mean(axis=1)
    pivot = pivot.sort_values("Avg MAE")

    medals = ["1st", "2nd", "3rd"] + ["" for _ in range(len(pivot) - 3)]
    pivot.insert(0, "Rank", medals[: len(pivot)])
    pivot = pivot.reset_index()

    def _highlight_best(s):
        """Color the minimum value in each numeric column green."""
        is_min = s == s.min()
        return ["font-weight:600; color:#3fb950" if v else "" for v in is_min]

    numeric_cols = ["Avg MAE"] + pos_cols
    fmt = {c: "{:.2f}" for c in numeric_cols}
    styled = (
        pivot.style
        .format(fmt)
        .apply(_highlight_best, subset=numeric_cols)
        .hide(axis="index")
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.divider()

    # MAE bar chart — primary metric for comparing models
    st.markdown('<span class="overline">Primary Metric</span>', unsafe_allow_html=True)
    st.markdown("### MAE by Position & Model")
    st.caption("Mean Absolute Error in PPR fantasy points. Lower is better.")

    fig_mae = px.bar(
        mdf.dropna(subset=["MAE"]),
        x="Position", y="MAE", color="Model",
        barmode="group", color_discrete_map=MDL_COLOR,
        labels={"MAE": "MAE (pts)", "Position": ""},
        category_orders={"Position": active_pos, "Model": active_mdl},
    )
    fig_mae.update_traces(marker_line_width=0, marker_opacity=0.9)
    fig_mae.update_layout(**chart(height=360))
    st.plotly_chart(fig_mae, use_container_width=True)

    col_l, col_r = st.columns(2, gap="medium")

    with col_l:
        st.markdown("### RMSE")
        st.caption("Penalises large errors more than MAE.")
        fig_rmse = px.bar(
            mdf.dropna(subset=["RMSE"]),
            x="Position", y="RMSE", color="Model",
            barmode="group", color_discrete_map=MDL_COLOR,
            labels={"RMSE": "RMSE (pts)", "Position": ""},
            category_orders={"Position": active_pos, "Model": active_mdl},
        )
        fig_rmse.update_traces(marker_line_width=0, marker_opacity=0.9)
        fig_rmse.update_layout(**chart(height=300, showlegend=False))
        st.plotly_chart(fig_rmse, use_container_width=True)

    with col_r:
        st.markdown("### R²")
        st.caption("Variance explained. 1.0 = perfect; 0 = predicts only the mean.")
        fig_r2 = px.bar(
            mdf.dropna(subset=["R2"]),
            x="Position", y="R2", color="Model",
            barmode="group", color_discrete_map=MDL_COLOR,
            labels={"R2": "R²", "Position": ""},
            category_orders={"Position": active_pos, "Model": active_mdl},
        )
        # Dotted zero line — makes it easy to see which models underperform the mean
        fig_r2.add_hline(y=0, line_dash="dot", line_color="#6e7681", line_width=1)
        fig_r2.update_traces(marker_line_width=0, marker_opacity=0.9)
        fig_r2.update_layout(**chart(height=300, showlegend=False))
        st.plotly_chart(fig_r2, use_container_width=True)

    st.divider()

    # CV vs. holdout — checks whether CV scores predicted real-world performance.
    # If holdout MAE is much higher than CV MAE, the model overfit the training years.
    st.markdown("### Cross-Validation vs. Holdout MAE")
    st.caption("Faded bars = 5-fold TimeSeriesSplit CV (2010–2023). Solid = 2024 holdout. Error bars ±1 std.")

    cv_df = mdf.dropna(subset=["CV_MAE", "MAE"])
    if not cv_df.empty:
        fig_cv = go.Figure()
        for mdl in active_mdl:
            sub = cv_df[cv_df["Model"] == mdl]
            if sub.empty:
                continue
            c = MDL_COLOR.get(mdl, TEXT_SEC)
            fig_cv.add_trace(go.Bar(
                name=mdl, x=sub["Position"], y=sub["CV_MAE"],
                marker_color=c, opacity=0.35,
                error_y=dict(type="data", array=sub["CV_Std"].fillna(0).tolist(),
                             color=c, thickness=1.5, width=5),
                legendgroup=mdl, showlegend=True,
            ))
            fig_cv.add_trace(go.Bar(
                name=mdl, x=sub["Position"], y=sub["MAE"],
                marker_color=c, opacity=0.9,
                legendgroup=mdl, showlegend=False,
            ))
        fig_cv.update_traces(marker_line_width=0)
        fig_cv.update_layout(**chart(barmode="group", height=340))
        st.plotly_chart(fig_cv, use_container_width=True)


# ============================================================
# TAB 2 — PREDICTIONS
# ============================================================
with tab2:
    st.markdown("### Predicted vs. Actual — 2024 Holdout")
    st.caption("Points are colored by absolute error: low error = position color, high error = red.")

    if preds_df is None:
        st.info("No predictions — run `python run_pipeline.py` first.")
        st.stop()

    c_pos, c_mod, _ = st.columns([1, 1, 2])
    s_pos = c_pos.selectbox("Position", active_pos, key="sc_pos")
    s_mod = c_mod.selectbox("Model",    active_mdl, key="sc_mod")

    sub = preds_df[(preds_df["position"] == s_pos) & (preds_df["model"] == s_mod)].copy()

    if sub.empty:
        st.info("No predictions for this selection.")
    else:
        sub["error"]    = (sub["predicted"] - sub["actual"]).abs()
        sub["residual"] = sub["predicted"] - sub["actual"]
        mae  = sub["error"].mean()
        bias = sub["residual"].mean()   # positive = model tends to over-predict
        over = (sub["predicted"] > sub["actual"]).mean() * 100

        # Actual vs. predicted scatter with a y=x perfect-prediction reference line
        max_val = max(sub["actual"].max(), sub["predicted"].max()) * 1.08
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val], mode="lines",
            line=dict(color="#6e7681", dash="dot", width=1.2),
            name="Perfect prediction", hoverinfo="skip",
        ))
        fig_sc.add_trace(go.Scatter(
            x=sub["actual"], y=sub["predicted"], mode="markers",
            marker=dict(
                color=sub["error"],
                # Color scale: position color at zero error fades to red at high error.
                # Capped at the 95th percentile so outliers don't wash out the scale.
                colorscale=[[0, POS_COLOR[s_pos]], [0.5, GOLD], [1, RED]],
                cmin=0, cmax=sub["error"].quantile(0.95),
                size=5, opacity=0.7, line=dict(width=0),
                colorbar=dict(title="Error (pts)", thickness=10,
                              len=0.65, tickfont=dict(size=10)),
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Week %{customdata[1]}<br>"
                "Actual: %{x:.1f} pts<br>"
                "Predicted: %{y:.1f} pts<br>"
                "Error: %{marker.color:.1f} pts<extra></extra>"
            ),
            customdata=sub[["player_name", "week"]].values
                if "player_name" in sub.columns else sub[["week", "week"]].values,
            name="Predictions",
        ))
        fig_sc.update_layout(
            **chart(height=440),
            xaxis_title="Actual PPR Points",
            yaxis_title="Predicted PPR Points",
            xaxis_range=[0, max_val],
            yaxis_range=[0, max_val],
        )
        st.plotly_chart(fig_sc, use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Mean Absolute Error",  f"{mae:.2f} pts")
        m2.metric("Prediction Bias",      f"{bias:+.2f} pts",
                  help="Positive = model over-predicts on average.")
        m3.metric("Over-predictions",     f"{over:.1f}%")

        st.divider()

        col_res, col_ov = st.columns(2, gap="medium")

        with col_res:
            st.markdown("### Residual Distribution")
            # A symmetric, zero-centered distribution indicates the model is unbiased.
            fig_res = px.histogram(
                sub, x="residual", nbins=40,
                color_discrete_sequence=[POS_COLOR[s_pos]],
                labels={"residual": "Predicted minus Actual (pts)"},
            )
            fig_res.add_vline(x=0,    line_dash="dot",   line_color="#6e7681")
            fig_res.add_vline(x=bias, line_dash="solid", line_color=GOLD,
                              annotation_text=f"bias {bias:+.1f}",
                              annotation_font_color=GOLD)
            fig_res.update_traces(marker_line_width=0, opacity=0.85)
            fig_res.update_layout(**chart(height=300))
            st.plotly_chart(fig_res, use_container_width=True)

        with col_ov:
            st.markdown("### Actual vs. Predicted Distribution")
            # Overlapping histograms show whether the model's predicted distribution
            # matches the shape of the actual score distribution.
            fig_ov = go.Figure()
            for label, series, color, opac in [
                ("Actual",    sub["actual"],    POS_COLOR[s_pos], 0.7),
                ("Predicted", sub["predicted"], "#6e7681",         0.55),
            ]:
                fig_ov.add_trace(go.Histogram(
                    x=series, name=label, nbinsx=35,
                    marker_color=color, opacity=opac, marker_line_width=0,
                ))
            fig_ov.update_layout(**chart(barmode="overlay", height=300))
            st.plotly_chart(fig_ov, use_container_width=True)

    # All-model side-by-side grid for quick visual comparison
    st.divider()
    st.markdown("### All Models — Side by Side")
    grid_pos = st.selectbox("Position", active_pos, key="grid_pos")
    gd = preds_df[(preds_df["position"] == grid_pos) & preds_df["model"].isin(active_mdl)]

    if not gd.empty:
        cols = st.columns(min(len(active_mdl), 4), gap="small")
        for col, mdl in zip(cols, active_mdl):
            g = gd[gd["model"] == mdl]
            if g.empty:
                continue
            mae_g = (g["actual"] - g["predicted"]).abs().mean()
            mv = max(g["actual"].max(), g["predicted"].max()) * 1.08
            fig_g = go.Figure()
            fig_g.add_trace(go.Scatter(
                x=[0, mv], y=[0, mv], mode="lines",
                line=dict(color="#6e7681", dash="dot", width=1),
                hoverinfo="skip", showlegend=False,
            ))
            fig_g.add_trace(go.Scatter(
                x=g["actual"], y=g["predicted"], mode="markers",
                marker=dict(color=MDL_COLOR.get(mdl, TEXT_SEC),
                            size=3.5, opacity=0.55, line=dict(width=0)),
                hoverinfo="skip", showlegend=False,
            ))
            fig_g.update_layout(
                **chart(height=230, margin=dict(t=42, b=28, l=28, r=12)),
                title=dict(
                    text=f"<b>{mdl}</b>   <span style='font-size:11px;color:{TEXT_SEC}'>MAE {mae_g:.2f} pts</span>",
                    font=dict(size=13, color=TEXT_PRI), x=0,
                ),
                xaxis_title="Actual", yaxis_title="Pred.",
                xaxis_range=[0, mv], yaxis_range=[0, mv],
            )
            col.plotly_chart(fig_g, use_container_width=True)


# ============================================================
# TAB 3 — EDA
# ============================================================
with tab3:
    st.markdown("### Exploratory Data Analysis")

    if features_df is None:
        st.info("No feature data — run `python run_pipeline.py` first.")
        st.stop()

    feat = features_df[features_df["position"].isin(active_pos)].copy()

    st.markdown('<span class="overline">Dataset Summary</span>', unsafe_allow_html=True)
    summary = (
        feat.groupby("position")[TARGET]
        .agg(Games="count", Mean="mean", Median="median",
             Std="std", Min="min", Max="max")
        .round(2)
        .reset_index()
        .rename(columns={"position": "Position"})
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.divider()

    st.markdown('<span class="overline">Scoring Distribution</span>', unsafe_allow_html=True)
    col_hist, col_box = st.columns(2, gap="medium")

    with col_hist:
        st.markdown("### PPR Score Histogram")
        fig_hist = px.histogram(
            feat, x=TARGET, color="position",
            color_discrete_map=POS_COLOR,
            barmode="overlay", opacity=0.75, nbins=55,
            labels={TARGET: "PPR Fantasy Points", "position": "Position"},
        )
        fig_hist.update_traces(marker_line_width=0)
        fig_hist.update_layout(**chart(height=330))
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_box:
        st.markdown("### Score Spread by Position")
        fig_box = px.box(
            feat, x="position", y=TARGET,
            color="position", color_discrete_map=POS_COLOR,
            labels={TARGET: "PPR Points", "position": ""},
            category_orders={"position": active_pos},
        )
        fig_box.update_traces(line_width=1.2, marker_size=3)
        fig_box.update_layout(**chart(height=330, showlegend=False))
        st.plotly_chart(fig_box, use_container_width=True)

    st.divider()

    st.markdown('<span class="overline">Historical Trend</span>', unsafe_allow_html=True)
    st.markdown("### Average Weekly PPR Points by Season")

    s_avg = (
        feat.groupby(["season", "position"])[TARGET]
        .mean().reset_index()
        .rename(columns={TARGET: "avg_ppr", "position": "Position"})
    )
    fig_trend = px.line(
        s_avg, x="season", y="avg_ppr", color="Position",
        color_discrete_map=POS_COLOR, markers=True,
        labels={"avg_ppr": "Avg PPR Points", "season": "Season"},
        category_orders={"Position": active_pos},
    )
    fig_trend.update_traces(marker=dict(size=6, line=dict(width=1.5, color="#0d1117")),
                            line_width=2)
    fig_trend.update_layout(**chart(height=360))
    st.plotly_chart(fig_trend, use_container_width=True)

    st.divider()

    # Correlation heatmap — shows which features are most linearly related to PPR score.
    # Selecting by position is important because the feature lists differ across positions.
    st.markdown('<span class="overline">Feature Correlations</span>', unsafe_allow_html=True)
    st.markdown("### Correlation Heatmap")

    hm_pos = st.selectbox("Position", active_pos, key="hm_pos")
    hm_cols = [f for f in POSITION_FEATURES.get(hm_pos, []) if f in feat.columns]
    hm_data = feat[feat["position"] == hm_pos][[TARGET] + hm_cols].copy()
    hm_data = hm_data.dropna(axis=1, how="all")
    hm_data = hm_data.fillna(hm_data.median(numeric_only=True))

    corr = hm_data.corr(numeric_only=True)
    fig_hm = px.imshow(
        corr, text_auto=".2f",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        aspect="auto",
    )
    fig_hm.update_traces(textfont_size=9)
    fig_hm.update_layout(
        **chart(height=520, margin=dict(t=16, b=16, l=16, r=16)),
        coloraxis_colorbar=dict(thickness=10, len=0.75, tickfont=dict(size=10)),
    )
    st.plotly_chart(fig_hm, use_container_width=True)


# ============================================================
# TAB 4 — FEATURE IMPORTANCE
# ============================================================
with tab4:
    st.markdown("### Random Forest Feature Importance")
    st.caption("Mean decrease in impurity. Higher score = stronger predictive signal.")

    if importances is None:
        st.info("No importance data — run `python run_pipeline.py` first.")
        st.stop()

    fi_pos = st.selectbox("Position", active_pos, key="fi_pos")
    fi_raw = importances.get(fi_pos, {})

    if fi_raw:
        fi_df = (
            pd.DataFrame(fi_raw.items(), columns=["Feature", "Importance"])
            .sort_values("Importance", ascending=True)
        )
        # Normalize to percentage so the y-axis reads as "share of total importance"
        # rather than raw impurity values, which are harder to interpret.
        fi_df["Pct"] = fi_df["Importance"] / fi_df["Importance"].sum() * 100

        fig_fi = px.bar(
            fi_df, x="Pct", y="Feature", orientation="h",
            labels={"Pct": "Relative Importance (%)", "Feature": ""},
            color="Pct",
            color_continuous_scale=[[0, SURFACE], [1, POS_COLOR[fi_pos]]],
        )
        fig_fi.update_traces(marker_line_width=0)
        fig_fi.update_layout(
            **chart(
                height=max(340, len(fi_df) * 28),
                margin=dict(t=16, b=16, l=160, r=16),
                showlegend=False,
            ),
            yaxis_tickfont_size=11,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info(f"No importance data for {fi_pos}.")

    st.divider()

    # Top-5 features across all positions at a glance
    st.markdown("### Top-5 Features by Position")
    p_cols = st.columns(len(active_pos), gap="medium")

    for col, pos in zip(p_cols, active_pos):
        pfi   = importances.get(pos, {})
        total = sum(pfi.values()) or 1
        col.markdown(
            f'<div style="display:inline-block;padding:3px 10px;border-radius:20px;'
            f'background:{POS_COLOR[pos]}22;border:1px solid {POS_COLOR[pos]}55;'
            f'color:{POS_COLOR[pos]};font-size:0.75rem;font-weight:600;'
            f'letter-spacing:0.06em;margin-bottom:10px">{pos}</div>',
            unsafe_allow_html=True,
        )
        if pfi:
            top5 = sorted(pfi.items(), key=lambda x: -x[1])[:5]
            rows = "".join(
                f'<div class="feat-row">'
                f'<span class="feat-name">{fn}</span>'
                f'<span class="feat-val">{v/total*100:.1f}%</span>'
                f'</div>'
                for fn, v in top5
            )
            col.markdown(rows, unsafe_allow_html=True)
        else:
            col.caption("No data")


# ============================================================
# TAB 5 — ROLLING WINDOW
# ============================================================
with tab5:
    st.markdown("### Rolling Window vs. Season Average")
    st.caption(
        f"Naive baseline: how well does each look-back window alone predict PPR points? "
        f"Holdout = {HOLDOUT_YEAR} season."
    )

    if features_df is None:
        st.info("No feature data — run `python run_pipeline.py` first.")
        st.stop()

    # For each position × window, compute the MAE of using that single feature
    # as the prediction. This reveals whether longer windows are more accurate,
    # and contextualizes how much the full models improve over a simple heuristic.
    rw_rows = []
    for pos in active_pos:
        hold = features_df[
            (features_df["position"] == pos) & (features_df["season"] == HOLDOUT_YEAR)
        ]
        for label, col_name in WINDOWS.items():
            if col_name not in hold.columns:
                continue
            valid = hold[[col_name, TARGET]].dropna()
            if len(valid) > 5:
                mae = (valid[TARGET] - valid[col_name]).abs().mean()
                rw_rows.append({"Position": pos, "Window": label, "MAE": round(mae, 3)})

    if not rw_rows:
        st.info("No holdout data for the rolling window analysis.")
        st.stop()

    rw_df = pd.DataFrame(rw_rows)
    window_order = list(WINDOWS.keys())

    fig_rw = px.line(
        rw_df, x="Window", y="MAE", color="Position",
        color_discrete_map=POS_COLOR, markers=True,
        labels={"MAE": "MAE (pts)", "Window": ""},
        category_orders={"Window": window_order, "Position": active_pos},
    )
    fig_rw.update_traces(
        marker=dict(size=9, line=dict(width=2, color="#0d1117")),
        line_width=2.2,
    )
    fig_rw.update_layout(**chart(height=380))
    st.plotly_chart(fig_rw, use_container_width=True)

    st.divider()

    st.markdown("### Breakdown by Position")
    p_cols = st.columns(len(active_pos), gap="small")
    for col, pos in zip(p_cols, active_pos):
        sub = rw_df[rw_df["Position"] == pos]
        if sub.empty:
            col.caption(f"{pos}: no data")
            continue
        best = sub.loc[sub["MAE"].idxmin(), "Window"]
        fig_m = px.bar(
            sub, x="Window", y="MAE",
            color_discrete_sequence=[POS_COLOR[pos]],
            labels={"MAE": "MAE (pts)", "Window": ""},
            category_orders={"Window": window_order},
        )
        fig_m.update_traces(marker_line_width=0, marker_opacity=0.85)
        fig_m.update_layout(
            **chart(height=230, margin=dict(t=40, b=52, l=32, r=8)),
            title=dict(
                text=f"<b style='color:{POS_COLOR[pos]}'>{pos}</b>",
                font=dict(size=13), x=0,
            ),
            xaxis_tickangle=-30,
            xaxis_tickfont_size=9.5,
        )
        col.plotly_chart(fig_m, use_container_width=True)
        col.markdown(
            f'<p style="text-align:center;font-size:0.75rem;color:{TEXT_SEC};margin-top:-12px">'
            f'Best: <span style="color:{TEXT_PRI}">{best}</span></p>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Heatmap — Position × Window MAE grid makes cross-position comparisons easy
    st.markdown("### MAE Heatmap — Position x Window")
    pivot = rw_df.pivot(index="Position", columns="Window", values="MAE")
    pivot = pivot[[w for w in window_order if w in pivot.columns]]
    fig_hm = px.imshow(
        pivot, text_auto=".2f",
        color_continuous_scale="RdYlGn_r",
        labels=dict(color="MAE (pts)"),
        aspect="auto",
    )
    fig_hm.update_traces(textfont_size=12)
    fig_hm.update_layout(
        **chart(height=260, margin=dict(t=16, b=16, l=60, r=16)),
        coloraxis_colorbar=dict(thickness=10, len=0.8, tickfont=dict(size=10)),
    )
    st.plotly_chart(fig_hm, use_container_width=True)


# --- Footer ---
st.divider()
st.markdown(
    f'<p style="text-align:center;color:#6e7681;font-size:0.75rem">'
    f'CSC 398 · Spring 2026 &nbsp;·&nbsp; '
    f'Justin Rzepko · Tiago Freitas · Jeremiah Trail &nbsp;·&nbsp; '
    f'Prof. Antonios / Martin</p>',
    unsafe_allow_html=True,
)
