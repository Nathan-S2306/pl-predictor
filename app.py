"""English football match predictor and season simulator (Premier League to League Two).

Run locally:  streamlit run app.py
"""
from __future__ import annotations

import datetime as dt
import html
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import charts
import model as M

DATA = Path(__file__).parent / "data"
# Changes whenever model.py is edited, so cached models and simulations are rebuilt after an update.
CODE_VERSION = Path(M.__file__).stat().st_mtime
LIVE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"

# Top to bottom of the pyramid. `zones` are the final-position ranges the simulation reports
# (1-based, inclusive; negative numbers count up from the bottom, so (-3, -1) is the bottom three).
LEAGUES = {
    "Premier League": {"code": "E0", "zones": {"Relegation": (-3, -1)}},
    "Championship": {"code": "E1", "zones": {"Auto promotion": (1, 2), "Play-offs": (3, 8), "Relegation": (-3, -1)}},
    "League One": {"code": "E2", "zones": {"Auto promotion": (1, 2), "Play-offs": (3, 6), "Relegation": (-4, -1)}},
    "League Two": {"code": "E3", "zones": {"Auto promotion": (1, 3), "Play-offs": (4, 7), "Relegation": (-2, -1)}},
}
LEAGUE_NAMES = list(LEAGUES)

st.set_page_config(page_title="English football predictor", page_icon="⚽", layout="centered",
                   initial_sidebar_state="collapsed")


# ------------------------------------------------------------------ data ----

@st.cache_data
def load_bundled() -> pd.DataFrame:
    """Premier League results shipped with the app (used if the live download fails)."""
    return M.clean_results(pd.read_csv(DATA / "results.csv"))


@st.cache_data
def load_schedule() -> pd.DataFrame:
    path = DATA / "fixtures.csv"
    if not path.exists():
        return pd.DataFrame(columns=["Date", "HomeTeam", "AwayTeam"])
    return pd.read_csv(path, parse_dates=["Date"])


@st.cache_data
def load_adjustments(league: str) -> dict[str, int]:
    """Points deductions (or awards) from data/adjustments.csv: League, Team, Points, Note."""
    path = DATA / "adjustments.csv"
    if not path.exists():
        return {}
    adj = pd.read_csv(path)
    adj = adj[adj["League"].str.strip() == league]
    return {str(t).strip(): int(p) for t, p in zip(adj["Team"], adj["Points"])}


@st.cache_data(ttl=6 * 3600, show_spinner="Checking for new results…")
def load_division(code: str, start_year: int) -> pd.DataFrame:
    """One division-season from football-data.co.uk (refreshed every 6 hours)."""
    season = f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"
    raw = pd.read_csv(LIVE_URL.format(season=season, code=code), encoding="latin-1", on_bad_lines="skip")
    frame = M.clean_results(raw.drop(columns=[c for c in raw.columns if c == "Season"]))
    frame["Season"] = f"{start_year}-{(start_year + 1) % 100:02d}"
    return frame


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_upcoming(code: str) -> pd.DataFrame:
    """Dated upcoming fixtures for one division from football-data.co.uk's fixtures file."""
    raw = pd.read_csv(FIXTURES_URL, encoding="utf-8-sig", on_bad_lines="skip")
    raw = raw.rename(columns={c: c.strip() for c in raw.columns})
    raw = raw[raw["Div"].astype(str).str.strip() == code]
    out = pd.DataFrame({"Date": pd.to_datetime(raw["Date"].astype(str).str.strip(), dayfirst=True, format="mixed", errors="coerce"),
                        "HomeTeam": raw["HomeTeam"].astype(str).str.strip(), "AwayTeam": raw["AwayTeam"].astype(str).str.strip()})
    return out.dropna(subset=["Date"])


def get_schedule(league: str, use_live: bool) -> pd.DataFrame:
    """Bundled fixture dates plus, when live data is on, the upcoming round from football-data.co.uk."""
    frames = [load_schedule()]
    if use_live:
        try:
            frames.append(load_upcoming(LEAGUES[league]["code"]))
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True)


def season_start(today: dt.date) -> int:
    return today.year if today.month >= 7 else today.year - 1


def load_live(league: str) -> pd.DataFrame:
    """Last season and this season for one league."""
    y = season_start(dt.date.today())
    code = LEAGUES[league]["code"]
    return pd.concat([load_division(code, y - 1), load_division(code, y)], ignore_index=True)


def came_down(league: str, results: pd.DataFrame) -> tuple[str, ...]:
    """Teams new to this league that played in the division above last season."""
    i = LEAGUE_NAMES.index(league)
    if i == 0:
        return ()
    try:
        above = load_division(LEAGUES[LEAGUE_NAMES[i - 1]]["code"], season_start(dt.date.today()) - 1)
    except Exception:
        return ()
    season = M.current_season(results)
    now = results[results["Season"] == season]
    before = results[results["Season"] != season]
    new = (set(now["HomeTeam"]) | set(now["AwayTeam"])) - (set(before["HomeTeam"]) | set(before["AwayTeam"]))
    return tuple(sorted(new & (set(above["HomeTeam"]) | set(above["AwayTeam"]))))


def get_results(league: str, use_live: bool, uploaded) -> tuple[pd.DataFrame | None, str]:
    bundled = load_bundled() if league == "Premier League" else None
    if uploaded is not None:
        try:
            return M.clean_results(pd.read_csv(uploaded)), "your uploaded file"
        except Exception as exc:  # show the problem, keep the app usable
            st.sidebar.error(f"Could not read the upload: {exc}")
    if use_live:
        try:
            live = load_live(league)
            if bundled is None:
                return live, "football-data.co.uk (live)"
            season = M.current_season(bundled)
            if (live["Season"] == season).sum() >= (bundled["Season"] == season).sum():
                return live, "football-data.co.uk (live)"
            st.sidebar.info("Live data is behind the bundled results, so the bundled file is used.")
        except Exception:
            if bundled is not None:
                st.sidebar.info("Live results could not be loaded, so the bundled file is used.")
    return bundled, "bundled results file"


@st.cache_resource(show_spinner=False)
def fit_model(results: pd.DataFrame, half_life: float, prior_sd: float, promoted: float,
              relegated: tuple[str, ...], code_version: float) -> M.GoalsModel:
    return M.fit(results, half_life_days=half_life, prior_sd=prior_sd, promoted_offset=promoted, relegated=relegated)


@st.cache_resource(show_spinner="Simulating the rest of the season…")
def run_simulation(_model: M.GoalsModel, results: pd.DataFrame, fixtures: pd.DataFrame, n_sims: int,
                   seed: int, uncertainty: bool, zones: dict, adjust: dict, key: tuple) -> M.SimulationResult:
    return M.simulate_season(_model, results, fixtures, n_sims=n_sims, seed=seed,
                             rating_uncertainty=uncertainty, zones=zones, points_adjust=adjust)


# -------------------------------------------------------------------- look ----

GREEN, CORAL, GOLD, MIST = "#0F5132", "#F0503F", "#E0A800", "#C5CFC6"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,400..900&display=swap');
.stApp, .stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp button, .stApp input,
.stApp [data-baseweb="select"] div, .pp { font-family: 'Archivo', system-ui, sans-serif; }
.block-container { padding-top: 4.4rem; max-width: 820px; }
.stApp h2, .stApp h3 { font-stretch: 112%; font-weight: 800; letter-spacing: -0.01em; }

.pp-hero { background: #0F5132; color: #F3F6F1; border-radius: 22px; padding: 22px 22px 18px; position: relative; overflow: hidden; }
.pp-hero::before { content: ""; position: absolute; right: -70px; top: 50%; width: 230px; height: 230px; margin-top: -115px;
  border: 3px solid rgba(243,246,241,.28); border-radius: 50%; }
.pp-hero::after { content: ""; position: absolute; right: 44px; top: 0; bottom: 0; border-left: 3px solid rgba(243,246,241,.28); }
.pp-hero h1 { font-family: 'Archivo', sans-serif; font-stretch: 125%; font-weight: 900; font-size: clamp(1.3rem, 5.4vw, 2.2rem);
  line-height: 1.02; letter-spacing: -0.02em; margin: 0 0 10px; padding: 0; color: #F3F6F1; max-width: 13em; position: relative; z-index: 1; }
.pp-hero p { margin: 0; font-size: .95rem; color: rgba(243,246,241,.86); max-width: 34em; position: relative; z-index: 1; }

/* tabs as pills (selectors cover both the current and older Streamlit tab markup) */
.stTabs [role="tablist"] { gap: 6px; border: 0 !important; box-shadow: none !important; flex-wrap: wrap; }
.stTabs .react-aria-SelectionIndicator, .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none !important; }
.stTabs [role="tab"] { border-radius: 999px; padding: 8px 16px; height: auto; background: #FFFFFF; border: 1.5px solid #D5DDD4; margin: 0; }
.stTabs [role="tab"] p { font-weight: 700; font-size: .95rem; color: #14201A; }
.stTabs [role="tab"][aria-selected="true"] { background: #14201A; border-color: #14201A; }
.stTabs [role="tab"][aria-selected="true"] p { color: #FFFFFF; }
.stTabs [role="tab"]:focus-visible { outline: 3px solid #E0A800; outline-offset: 2px; }

.pp-card { background: #FFFFFF; border: 1.5px solid #DCE3DB; border-radius: 18px; padding: 20px; }
.pp-score { display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; gap: 12px; }
.pp-score .team { font-stretch: 112%; font-weight: 800; font-size: clamp(1rem, 3.6vw, 1.45rem); line-height: 1.1; }
.pp-score .team.away { text-align: right; }
.pp-score .team small { display: block; font-stretch: 100%; font-weight: 500; font-size: .8rem; color: #5B6A60; margin-top: 4px; }
.pp-score .num { font-stretch: 125%; font-weight: 900; font-size: clamp(2.4rem, 11vw, 4.2rem); letter-spacing: -0.03em; line-height: 1;
  font-variant-numeric: tabular-nums; white-space: nowrap; }
.pp-bar { display: flex; height: 14px; border-radius: 999px; overflow: hidden; gap: 2px; margin: 18px 0 8px; }
.pp-bar span { display: block; height: 100%; }
.pp-bar.mini { height: 8px; margin: 0; min-width: 90px; }
.pp-legend { display: flex; justify-content: space-between; font-size: .88rem; font-weight: 600; gap: 8px; }
.pp-legend b { font-weight: 800; font-variant-numeric: tabular-nums; }
.pp-legend i { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 6px; }
.pp-facts { display: flex; flex-wrap: wrap; gap: 8px 22px; margin-top: 16px; padding-top: 14px; border-top: 1.5px solid #E6ECE5; font-size: .9rem; color: #3D4A42; }
.pp-facts b { color: #14201A; font-variant-numeric: tabular-nums; }

.pp-list { display: grid; gap: 6px; }
.pp-fx { display: grid; grid-template-columns: 74px 1fr 54px 1fr 110px; align-items: center; gap: 10px; background: #FFFFFF;
  border: 1.5px solid #DCE3DB; border-radius: 12px; padding: 9px 12px; font-size: .92rem; }
.pp-fx .d { color: #5B6A60; font-size: .8rem; }
.pp-fx .h { text-align: right; font-weight: 600; }
.pp-fx .a { font-weight: 600; }
.pp-fx .me { font-weight: 800; }
.pp-fx .s { text-align: center; font-weight: 800; background: #14201A; color: #FFFFFF; border-radius: 8px; padding: 3px 0; font-variant-numeric: tabular-nums; }
.pp-fx .s.sim { background: #E0A800; color: #14201A; }
@media (max-width: 600px) {
  .pp-fx { grid-template-columns: 1fr 50px 1fr; }
  .pp-fx .d { grid-column: 1 / -1; }
  .pp-fx .pp-bar { grid-column: 1 / -1; }
}

.pp-tablewrap { overflow-x: auto; background: #FFFFFF; border: 1.5px solid #DCE3DB; border-radius: 18px; }
table.pp-table { border-collapse: collapse; width: 100%; font-size: .9rem; font-variant-numeric: tabular-nums; }
.pp-table th { text-align: right; font-weight: 600; color: #5B6A60; font-size: .78rem; padding: 12px 10px 8px; white-space: nowrap; }
.pp-table th.l, .pp-table td.l { text-align: left; }
.pp-table td { padding: 8px 10px; border-top: 1.5px solid #EDF1EC; text-align: right; white-space: nowrap; }
.pp-table td.team { font-weight: 700; }
.pp-table td small { color: #5B6A60; font-weight: 500; }
.pp-table tr.me td { background: #FFF6D6; }
.pp-table tr.cut td { border-top: 2.5px dashed #9AA79C; }
.pp-chance { display: inline-grid; grid-template-columns: 44px 38px; align-items: center; gap: 6px; }
.pp-chance span { display: block; height: 8px; border-radius: 999px; background: #EDF1EC; overflow: hidden; }
.pp-chance span i { display: block; height: 100%; border-radius: 999px; }
.pp-scores { display: grid; gap: 7px; }
.pp-scores div { display: grid; grid-template-columns: 44px 1fr 48px; gap: 10px; align-items: center; font-variant-numeric: tabular-nums; }
.pp-scores b { font-weight: 800; }
.pp-scores span { display: block; height: 10px; border-radius: 999px; background: #0F5132; }
.pp-scores em { font-style: normal; text-align: right; color: #3D4A42; font-size: .88rem; }
.pp-note { color: #5B6A60; font-size: .86rem; max-width: 62ch; }
</style>
"""


def pct(x: float) -> str:
    return f"{x:.0%}" if x >= 0.095 or x == 0 else ("<1%" if x < 0.005 else f"{x:.1%}")


def esc(s) -> str:
    return html.escape(str(s))


def show(fig) -> None:
    st.pyplot(fig)
    plt.close(fig)


def bar3(p_home: float, p_draw: float, p_away: float, mini: bool = False) -> str:
    label = f"Home win {p_home:.0%}, draw {p_draw:.0%}, away win {p_away:.0%}"
    return (f'<div class="pp-bar{" mini" if mini else ""}" role="img" aria-label="{label}" title="{label}">'
            f'<span style="width:{p_home:.1%};background:{GREEN}"></span>'
            f'<span style="width:{p_draw:.1%};background:{MIST}"></span>'
            f'<span style="width:{p_away:.1%};background:{CORAL}"></span></div>')


def fixture_list(fx: pd.DataFrame, me: str | None = None, scores: list | None = None) -> str:
    """Rows of fixtures with a score chip (predicted, or simulated when `scores` is given) and a result bar."""
    rows = []
    for k, r in enumerate(fx.itertuples()):
        p = gm.predict(r.HomeTeam, r.AwayTeam)
        sc = scores[k] if scores else p["result_score"]
        date = r.Date.strftime("%a %d %b") if pd.notna(r.Date) else "Date TBC"
        rows.append(
            f'<div class="pp-fx"><span class="d">{date}</span>'
            f'<span class="h{" me" if r.HomeTeam == me else ""}">{esc(r.HomeTeam)}</span>'
            f'<span class="s{" sim" if scores else ""}">{sc[0]}–{sc[1]}</span>'
            f'<span class="a{" me" if r.AwayTeam == me else ""}">{esc(r.AwayTeam)}</span>'
            f'{bar3(p["p_home"], p["p_draw"], p["p_away"], mini=True)}</div>')
    return f'<div class="pp pp-list">{"".join(rows)}</div>'


def chance_cell(p: float, colour: str) -> str:
    return (f'<td><div class="pp-chance"><span><i style="width:{max(p, 0.0):.1%};background:{colour}"></i></span>'
            f'{pct(p)}</div></td>')


def projected_table(s: pd.DataFrame, chance_cols: list[str], me: str | None, cuts: set[int]) -> str:
    colours = {"Title": GOLD, "Play-offs": GOLD, "Relegation": CORAL}
    head = ('<tr><th>#</th><th class="l">Team</th><th>Played</th><th>Pts</th><th>Projected</th><th>Goals</th>'
            + "".join(f"<th>{esc(c)}</th>" for c in chance_cols) + "</tr>")
    body = []
    for r in s.itertuples():
        cls = " ".join(c for c, on in (("me", r.Team == me), ("cut", r.Pos in cuts)) if on)
        cells = "".join(chance_cell(float(s.at[r.Index, c]), colours.get(c, GREEN)) for c in chance_cols)
        body.append(
            f'<tr class="{cls}"><td>{r.Pos}</td><td class="l team">{esc(r.Team)}</td><td>{r.Played}</td>'
            f'<td>{int(s.at[r.Index, "Pts now"])}</td>'
            f'<td><b>{s.at[r.Index, "Exp pts"]:.0f}</b> <small>{s.at[r.Index, "Pts low"]:.0f}–{s.at[r.Index, "Pts high"]:.0f}</small></td>'
            f'<td>{s.at[r.Index, "Exp GF"]:.0f}–{s.at[r.Index, "Exp GA"]:.0f}</td>{cells}</tr>')
    return f'<div class="pp pp-tablewrap"><table class="pp-table">{head}{"".join(body)}</table></div>'


# ---------------------------------------------------------------- settings ----

with st.sidebar:
    st.header("Settings")
    st.subheader("Data")
    use_live = st.toggle("Fetch latest results automatically", value=True,
                         help="Pulls results from football-data.co.uk. Falls back to the bundled file if that fails.")
    uploaded = st.file_uploader("Or upload results (CSV)", type="csv",
                                help="Results for the selected league. Columns: Date, HomeTeam, AwayTeam, FTHG, FTAG. "
                                     "A football-data.co.uk file works as is.")
    st.subheader("Model")
    half_life = st.slider("Form half-life (days)", 60, 730, 365, step=5,
                          help="A match this old counts half as much as one played today.")
    prior_sd = st.slider("Rating spread", 0.10, 0.60, 0.30, step=0.05,
                         help="Lower values pull every team towards league average; higher values trust results more.")
    promoted = st.slider("New-team adjustment", 0.0, 0.5, 0.20, step=0.05,
                         help="Teams new to the division have no history in it. Promoted teams start this far below "
                              "average and relegated teams this far above, before their results are counted.")
    st.subheader("Simulation")
    n_sims = st.select_slider("Simulated seasons", options=[1_000, 5_000, 10_000, 20_000, 50_000], value=10_000)
    uncertainty = st.toggle("Allow for rating uncertainty", value=True,
                            help="Each simulated season draws slightly different team ratings, which widens the range of outcomes.")
    top_n = st.selectbox("Premier League European places (top N)", [4, 5, 6, 7], index=0)
    seed = st.number_input("Random seed", value=42, step=1)

# ------------------------------------------------------------------ header ----

st.html(CSS)
hero = st.empty()
league = st.segmented_control("League", LEAGUE_NAMES, default=LEAGUE_NAMES[0], label_visibility="collapsed") or LEAGUE_NAMES[0]
if league != "Premier League":
    top_n = 0

results, source = get_results(league, use_live, uploaded)
if results is None:
    hero.html('<div class="pp pp-hero"><h1>Predict the weekend.<br>Simulate the season.</h1></div>')
    st.error(f"{league} results could not be downloaded from football-data.co.uk. Try again shortly, or upload a "
             f"results file for this league under Settings (file {LEAGUES[league]['code']}.csv from football-data.co.uk).")
    st.stop()
season = M.current_season(results)
current = results[results["Season"] == season]
relegated_in = came_down(league, results) if source.startswith("football-data") else ()
gm = fit_model(results, float(half_life), float(prior_sd), float(promoted), relegated_in, CODE_VERSION)
fixtures = M.remaining_fixtures(results, get_schedule(league, use_live))
teams = sorted(set(current["HomeTeam"]) | set(current["AwayTeam"]))
adjust = {t: p for t, p in load_adjustments(league).items() if t in teams}
n_teams = len(teams)
zones = ({f"Top {top_n}": (1, int(top_n))} if top_n else {}) | {
    name: (lo if lo > 0 else n_teams + 1 + lo, hi if hi > 0 else n_teams + 1 + hi)
    for name, (lo, hi) in LEAGUES[league]["zones"].items()}

hero.html(f'<div class="pp pp-hero"><h1>Predict the weekend.<br>Simulate the season.</h1>'
          f'<p>{esc(league)} {season}: {len(current)} matches played, {len(fixtures)} to go. '
          f'Results to {results["Date"].max():%-d %B} from {esc(source)}.</p></div>')

NO_TEAM = "No team"
follow = st.selectbox("Follow a team", [NO_TEAM, *teams], key=f"follow_{league}",
                      help="Highlights your team in the table and opens its fixtures and points forecast first.")
me = None if follow == NO_TEAM else follow

tab_match, tab_next, tab_table, tab_ratings, tab_data = st.tabs(["Match", "Fixtures", "Table", "Ratings", "Data"])

# ------------------------------------------------------------------- match ----

with tab_match:
    c1, c2 = st.columns(2)
    home = c1.selectbox("Home", teams, index=teams.index(me) if me else 0)
    away_options = [t for t in teams if t != home]
    away = c2.selectbox("Away", away_options, index=0)
    neutral = st.checkbox("Neutral venue", value=False)
    pr = gm.predict(home, away, neutral=neutral)
    rs = pr["result_score"]
    st.html(
        f'<div class="pp pp-card"><div class="pp-score">'
        f'<div class="team">{esc(home)}<small>expected goals {pr["home_xg"]:.2f}</small></div>'
        f'<div class="num">{rs[0]}–{rs[1]}</div>'
        f'<div class="team away">{esc(away)}<small>expected goals {pr["away_xg"]:.2f}</small></div></div>'
        f'{bar3(pr["p_home"], pr["p_draw"], pr["p_away"])}'
        f'<div class="pp-legend"><span><i style="background:{GREEN}"></i>{esc(home)} <b>{pct(pr["p_home"])}</b></span>'
        f'<span><i style="background:{MIST}"></i>Draw <b>{pct(pr["p_draw"])}</b></span>'
        f'<span><i style="background:{CORAL}"></i>{esc(away)} <b>{pct(pr["p_away"])}</b></span></div>'
        f'<div class="pp-facts"><span>This exact score <b>{pct(float(pr["matrix"][rs]))}</b></span>'
        f'<span>Over 2.5 goals <b>{pct(pr["p_over_2_5"])}</b></span>'
        f'<span>Both teams score <b>{pct(pr["p_btts"])}</b></span></div></div>')

    st.subheader("Most likely scores")
    P = pr["matrix"]
    flat = sorted(((P[i, j], i, j) for i in range(P.shape[0]) for j in range(P.shape[1])), reverse=True)[:8]
    top = flat[0][0]
    st.html('<div class="pp pp-card"><div class="pp-scores">' + "".join(
        f'<div><b>{i}–{j}</b><span style="width:{p / top:.0%}"></span><em>{p:.1%}</em></div>' for p, i, j in flat)
        + "</div></div>")
    st.subheader("Every score")
    show(charts.scoreline_heatmap(P, home, away))

# ---------------------------------------------------------------- fixtures ----

with tab_next:
    dated = fixtures.dropna(subset=["Date"])
    if len(dated):
        first = dated["Date"].min()
        st.subheader("Next round")
        st.html(fixture_list(dated[dated["Date"] <= first + pd.Timedelta(days=4)], me))
    team = st.selectbox("Remaining fixtures for", teams, index=teams.index(me) if me else 0, key=f"fx_{league}")
    mine = fixtures[(fixtures["HomeTeam"] == team) | (fixtures["AwayTeam"] == team)]
    mode = st.segmented_control("Scores", ["Predicted", "Play it out"], default="Predicted", label_visibility="collapsed") or "Predicted"
    if mode == "Predicted":
        st.html(fixture_list(mine, team))
        st.html('<p class="pp pp-note">The dark chip is the most likely score for the most likely result, and the bar '
                'shows home win, draw and away win chances. Single predicted scores always look low and alike; '
                'switch to “Play it out” for a realistic run of results.</p>')
    else:
        if st.button("Play again"):
            st.session_state["run"] = st.session_state.get("run", 0) + 1
        run = st.session_state.get("run", 0)
        scores = M.sample_scores(gm, mine, seed=1000 * run + teams.index(team))
        is_home = (mine["HomeTeam"] == team).to_numpy()
        gf = sum(h if hm else a for (h, a), hm in zip(scores, is_home))
        ga = sum(a if hm else h for (h, a), hm in zip(scores, is_home))
        w = sum((h > a) if hm else (a > h) for (h, a), hm in zip(scores, is_home))
        d = sum(h == a for h, a in scores)
        st.markdown(f"**In this run {team} win {w}, draw {d} and lose {len(scores) - w - d}**, scoring {gf} and "
                    f"conceding {ga}: {3 * w + d} points from {len(scores)} games.")
        st.html(fixture_list(mine, team, scores))
        st.html('<p class="pp pp-note">One random run drawn from the model’s score chances. It is one possible future, '
                'not a forecast; the Table tab averages thousands of them.</p>')

# ------------------------------------------------------------------- table ----

with tab_table:
    sim_key = (league, len(results), str(results["Date"].max()), half_life, prior_sd, promoted, source, CODE_VERSION)
    sim = run_simulation(gm, results, fixtures, int(n_sims), int(seed), bool(uncertainty), zones, adjust, sim_key)
    s = sim.summary
    chance_cols = ["Title", *zones]
    leader = s.iloc[0]
    st.subheader(f"{leader['Team']} finish top in {pct(leader['Title'])} of {sim.n_sims:,} simulated seasons")
    cuts = {hi + 1 for name, (lo, hi) in zones.items() if name != "Relegation"} | {
        lo for name, (lo, hi) in zones.items() if name == "Relegation"}
    st.html(projected_table(s, chance_cols, me, cuts))
    note = ("Projected is the average final points, with the range nine seasons in ten fall inside. "
            "Goals are projected for and against. Dashed lines mark the "
            + " and ".join(filter(None, [", ".join(n.lower() for n in list(zones)[:-1]), list(zones)[-1].lower()])) + " places.")
    if adjust:
        note += " Points adjustments applied: " + ", ".join(f"{t} {p:+d}" for t, p in adjust.items()) + "."
    st.html(f'<p class="pp pp-note">{esc(note)}</p>')

    st.subheader("Where each team could finish")
    show(charts.position_heatmap(sim.position_probs, highlight=me))

    st.subheader("Final points")
    order = list(s["Team"])
    pick = st.selectbox("Team", order, index=order.index(me) if me else 0, key=f"pts_{league}")
    col = sim.teams.index(pick)
    show(charts.points_histogram(sim.points[:, col].astype(int), pick, int(s.loc[s["Team"] == pick, "Pts now"].iloc[0])))
    st.download_button("Download projected table (CSV)", s.to_csv(index=False), "projected_table.csv", "text/csv")

# ----------------------------------------------------------------- ratings ----

with tab_ratings:
    st.subheader("How strong the model thinks each team is")
    r = gm.ratings()
    r = r[r["Team"].isin(teams)].reset_index(drop=True)
    r.insert(0, "Rank", range(1, len(r) + 1))
    st.dataframe(r[["Rank", "Team", "Attack", "Defence"]], hide_index=True, height=35 * (len(r) + 1) + 3,
                 column_config={
                     "Attack": st.column_config.NumberColumn("Attack", format="%.2f", help="Goals scored relative to an average team (1.00). Higher is better."),
                     "Defence": st.column_config.NumberColumn("Defence", format="%.2f", help="Goals conceded relative to an average team (1.00). Lower is better.")})
    notes = [f"Attack is goals scored and defence is goals conceded, each relative to an average team at 1.00, so a low "
             f"defence number is good. Home teams score {np.exp(gm.home_adv) - 1:.0%} more. Based on {gm.fitted_on} matches."]
    if gm.promoted:
        notes.append("Treated as newly promoted: " + ", ".join(gm.promoted) + ".")
    if gm.relegated:
        notes.append("Treated as relegated from the division above: " + ", ".join(gm.relegated) + ".")
    st.html(f'<p class="pp pp-note">{esc(" ".join(notes))}</p>')

# -------------------------------------------------------------------- data ----

with tab_data:
    st.subheader(f"{season} table so far")
    st.dataframe(M.league_table(current, points_adjust=adjust).reset_index(), hide_index=True, height=35 * (len(teams) + 1) + 3)
    st.subheader("Results the model uses")
    shown = results.sort_values("Date", ascending=False).copy()
    shown["Date"] = shown["Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(shown, hide_index=True)
    st.download_button("Download results (CSV)", shown.to_csv(index=False), "results.csv", "text/csv")

st.html('<p class="pp pp-note">Model and simulation settings are in the menu at the top left.</p>')
