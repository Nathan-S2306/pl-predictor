"""Premier League match predictor and season simulator.

Run locally:  streamlit run app.py
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import charts
import model as M

DATA = Path(__file__).parent / "data"
LIVE_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"

st.set_page_config(page_title="Premier League predictor", page_icon="⚽", layout="wide")


# ------------------------------------------------------------------ data ----

@st.cache_data
def load_bundled() -> pd.DataFrame:
    return M.clean_results(pd.read_csv(DATA / "results.csv"))


@st.cache_data
def load_schedule() -> pd.DataFrame:
    path = DATA / "fixtures.csv"
    if not path.exists():
        return pd.DataFrame(columns=["Date", "HomeTeam", "AwayTeam"])
    return pd.read_csv(path, parse_dates=["Date"])


@st.cache_data(ttl=6 * 3600, show_spinner="Checking for new results…")
def load_live(today: dt.date) -> pd.DataFrame:
    """Last season and this season from football-data.co.uk (refreshed every 6 hours)."""
    start = today.year if today.month >= 7 else today.year - 1
    frames = []
    for y in (start - 1, start):
        code = f"{y % 100:02d}{(y + 1) % 100:02d}"
        raw = pd.read_csv(LIVE_URL.format(code=code), encoding="latin-1", on_bad_lines="skip")
        frame = M.clean_results(raw.drop(columns=[c for c in raw.columns if c == "Season"]))
        frame["Season"] = f"{y}-{(y + 1) % 100:02d}"
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def get_results(use_live: bool, uploaded) -> tuple[pd.DataFrame, str]:
    bundled = load_bundled()
    if uploaded is not None:
        try:
            return M.clean_results(pd.read_csv(uploaded)), "your uploaded file"
        except Exception as exc:  # show the problem, keep the app usable
            st.sidebar.error(f"Could not read the upload: {exc}")
    if use_live:
        try:
            live = load_live(dt.date.today())
            season = M.current_season(bundled)
            if (live["Season"] == season).sum() >= (bundled["Season"] == season).sum():
                return live, "football-data.co.uk (live)"
            st.sidebar.info("Live data is behind the bundled results, so the bundled file is used.")
        except Exception:
            st.sidebar.info("Live results could not be loaded, so the bundled file is used.")
    return bundled, "bundled results file"


@st.cache_resource(show_spinner=False)
def fit_model(results: pd.DataFrame, half_life: float, prior_sd: float, promoted: float) -> M.GoalsModel:
    return M.fit(results, half_life_days=half_life, prior_sd=prior_sd, promoted_offset=promoted)


@st.cache_resource(show_spinner="Simulating the rest of the season…")
def run_simulation(_model: M.GoalsModel, results: pd.DataFrame, fixtures: pd.DataFrame, n_sims: int,
                   seed: int, uncertainty: bool, top_n: int, key: tuple) -> M.SimulationResult:
    return M.simulate_season(_model, results, fixtures, n_sims=n_sims, seed=seed,
                             rating_uncertainty=uncertainty, top_n=top_n)


def pct(x: float) -> str:
    return f"{x:.0%}" if x >= 0.095 or x == 0 else ("<1%" if x < 0.005 else f"{x:.1%}")


def show(fig) -> None:
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------------------- sidebar ----

with st.sidebar:
    st.header("Data")
    use_live = st.toggle("Fetch latest results automatically", value=True,
                         help="Pulls results from football-data.co.uk. Falls back to the bundled file if that fails.")
    uploaded = st.file_uploader("Or upload results (CSV)", type="csv",
                                help="Columns: Date, HomeTeam, AwayTeam, FTHG, FTAG. A football-data.co.uk file works as is.")
    st.header("Model")
    half_life = st.slider("Form half-life (days)", 60, 730, 365, step=5,
                          help="A match this old counts half as much as one played today.")
    prior_sd = st.slider("Rating spread", 0.10, 0.60, 0.30, step=0.05,
                         help="Lower values pull every team towards league average; higher values trust results more.")
    promoted = st.slider("Promoted-team handicap", 0.0, 0.5, 0.20, step=0.05,
                         help="How far below average newly promoted teams start, before their results are counted.")
    st.header("Simulation")
    n_sims = st.select_slider("Simulated seasons", options=[1_000, 5_000, 10_000, 20_000, 50_000], value=10_000)
    uncertainty = st.toggle("Allow for rating uncertainty", value=True,
                            help="Each simulated season draws slightly different team ratings, which widens the range of outcomes.")
    top_n = st.selectbox("European places (top N)", [4, 5, 6, 7], index=0)
    seed = st.number_input("Random seed", value=42, step=1)

results, source = get_results(use_live, uploaded)
season = M.current_season(results)
current = results[results["Season"] == season]
gm = fit_model(results, float(half_life), float(prior_sd), float(promoted))
fixtures = M.remaining_fixtures(results, load_schedule())
teams = sorted(set(current["HomeTeam"]) | set(current["AwayTeam"]))

st.title("Premier League predictor")
st.caption(f"{season} season · {len(current)} matches played, {len(fixtures)} to go · "
           f"results to {results['Date'].max():%d %b %Y} from {source}")

tab_match, tab_next, tab_table, tab_ratings, tab_data = st.tabs(
    ["Match predictor", "Fixtures", "Season simulation", "Team ratings", "Data"])

# ------------------------------------------------------- match predictor ----

with tab_match:
    c1, c2, c3 = st.columns([2, 2, 1])
    home = c1.selectbox("Home team", teams, index=0)
    away_options = [t for t in teams if t != home]
    away = c2.selectbox("Away team", away_options, index=0)
    neutral = c3.checkbox("Neutral venue", value=False)
    pr = gm.predict(home, away, neutral=neutral)

    m1, m2, m3 = st.columns(3)
    m1.metric(f"{home} win", pct(pr["p_home"]))
    m2.metric("Draw", pct(pr["p_draw"]))
    m3.metric(f"{away} win", pct(pr["p_away"]))
    m4, m5, m6 = st.columns(3)
    m4.metric("Expected goals", f"{pr['home_xg']:.2f} – {pr['away_xg']:.2f}")
    m5.metric("Most likely score", f"{pr['likely_score'][0]}–{pr['likely_score'][1]}", pct(pr["likely_score_p"]), delta_color="off")
    m6.metric("Over 2.5 goals / both score", f"{pct(pr['p_over_2_5'])} / {pct(pr['p_btts'])}")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Chance of each score")
        show(charts.scoreline_heatmap(pr["matrix"], home, away))
    with right:
        st.subheader("Most likely scores")
        P = pr["matrix"]
        flat = sorted(((P[i, j], i, j) for i in range(P.shape[0]) for j in range(P.shape[1])), reverse=True)[:10]
        st.dataframe(pd.DataFrame({"Score": [f"{i}–{j}" for _, i, j in flat],
                                   "Chance": [p * 100 for p, _, _ in flat]}),
                     hide_index=True,
                     column_config={"Chance": st.column_config.NumberColumn(format="%.1f%%")})

# -------------------------------------------------------------- fixtures ----


def predict_fixtures(fx: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in fx.itertuples():
        p = gm.predict(r.HomeTeam, r.AwayTeam)
        rows.append({"Date": r.Date.strftime("%a %d %b") if pd.notna(r.Date) else "TBC",
                     "Home": r.HomeTeam, "Away": r.AwayTeam,
                     "Predicted": f"{p['likely_score'][0]}–{p['likely_score'][1]}",
                     "xG": f"{p['home_xg']:.1f} – {p['away_xg']:.1f}",
                     "Home win": p["p_home"] * 100, "Draw": p["p_draw"] * 100, "Away win": p["p_away"] * 100})
    return pd.DataFrame(rows)


PCT_COLS = {c: st.column_config.ProgressColumn(c, format="%.0f%%", min_value=0, max_value=100)
            for c in ("Home win", "Draw", "Away win")}

with tab_next:
    dated = fixtures.dropna(subset=["Date"])
    if len(dated):
        st.subheader("Next scheduled fixtures")
        st.dataframe(predict_fixtures(dated.head(10)), hide_index=True, column_config=PCT_COLS)
    else:
        st.info("No dated fixtures are loaded. Add upcoming dates to data/fixtures.csv, or browse by team below.")
    st.subheader("Remaining fixtures by team")
    team = st.selectbox("Team", teams, key="fixtures_team")
    mine = fixtures[(fixtures["HomeTeam"] == team) | (fixtures["AwayTeam"] == team)]
    st.dataframe(predict_fixtures(mine), hide_index=True, column_config=PCT_COLS)
    st.caption("“Predicted” is the single most likely score. Dates show TBC unless they are listed in data/fixtures.csv.")

# ------------------------------------------------------------- simulation ----

with tab_table:
    sim_key = (len(results), str(results["Date"].max()), half_life, prior_sd, promoted, source)
    sim = run_simulation(gm, results, fixtures, int(n_sims), int(seed), bool(uncertainty), int(top_n), sim_key)
    s = sim.summary
    top_col = f"Top {top_n}"
    leader = s.iloc[0]
    k1, k2, k3 = st.columns(3)
    k1.metric("Title favourite", leader["Team"], pct(s["Title"].max()), delta_color="off")
    k2.metric("Projected winning total", f"{leader['Exp pts']:.0f} pts")
    k3.metric("Simulated seasons", f"{sim.n_sims:,}")

    st.subheader("Projected final table")
    view = s[["Pos", "Team", "Played", "Pts now", "Exp pts", "Pts low", "Pts high", "Title", top_col, "Relegation"]].copy()
    view["Range"] = view["Pts low"].round().astype(int).astype(str) + "–" + view["Pts high"].round().astype(int).astype(str)
    for c in ("Title", top_col, "Relegation"):
        view[c] = view[c] * 100
    view = view[["Pos", "Team", "Played", "Pts now", "Exp pts", "Range", "Title", top_col, "Relegation"]]
    st.dataframe(view, hide_index=True, height=38 * (len(view) + 1),
                 column_config={
                     "Exp pts": st.column_config.NumberColumn("Projected pts", format="%.1f"),
                     "Range": st.column_config.TextColumn("90% range", help="Nine simulations in ten finish inside this points range."),
                     **{c: st.column_config.ProgressColumn(c, format="%.1f%%", min_value=0, max_value=100)
                        for c in ("Title", top_col, "Relegation")}})

    st.subheader("Chance of each finishing position")
    show(charts.position_heatmap(sim.position_probs))

    st.subheader("Final points for one team")
    pick = st.selectbox("Team", list(s["Team"]), key="points_team")
    col = sim.teams.index(pick)
    show(charts.points_histogram(sim.points[:, col].astype(int), pick, int(s.loc[s["Team"] == pick, "Pts now"].iloc[0])))
    st.download_button("Download projected table (CSV)", s.to_csv(index=False), "projected_table.csv", "text/csv")

# ---------------------------------------------------------------- ratings ----

with tab_ratings:
    st.subheader("Team ratings")
    r = gm.ratings()
    r = r[r["Team"].isin(teams)].reset_index(drop=True)
    r.insert(0, "Rank", range(1, len(r) + 1))
    st.dataframe(r[["Rank", "Team", "Attack", "Defence"]], hide_index=True, height=38 * (len(r) + 1),
                 column_config={
                     "Attack": st.column_config.NumberColumn("Attack", format="%.2f", help="Goals scored relative to an average team (1.00). Higher is better."),
                     "Defence": st.column_config.NumberColumn("Defence", format="%.2f", help="Goals conceded relative to an average team (1.00). Lower is better.")})
    a, b, c = st.columns(3)
    a.metric("Home advantage", f"+{np.exp(gm.home_adv) - 1:.0%} goals")
    b.metric("Matches used", f"{gm.fitted_on}")
    c.metric("Low-score adjustment (rho)", f"{gm.rho:+.3f}")
    if gm.promoted:
        st.caption("Treated as newly promoted: " + ", ".join(gm.promoted) + ".")

# ------------------------------------------------------------------- data ----

with tab_data:
    st.subheader(f"{season} table")
    st.dataframe(M.league_table(current).reset_index(), hide_index=True, height=38 * (len(teams) + 1))
    st.subheader("Results used by the model")
    shown = results.sort_values("Date", ascending=False).copy()
    shown["Date"] = shown["Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(shown, hide_index=True)
    st.download_button("Download results (CSV)", shown.to_csv(index=False), "results.csv", "text/csv")
