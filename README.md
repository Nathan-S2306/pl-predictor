# English football predictor

A Streamlit app that predicts matches (goals and results) in the Premier League,
Championship, League One and League Two, and simulates the rest of the season to
project each final table. Pick the league in the sidebar.

## What it does

- **Match predictor** – pick any two teams: expected goals, win/draw/win
  chances, the most likely scores and a full scoreline grid.
- **Fixtures** – predictions for the next scheduled round and for any team's
  remaining games.
- **Season simulation** – Monte Carlo simulation of every remaining fixture
  (10,000 seasons by default): projected points, title, top-N and relegation
  chances, and the chance of each finishing position.
- **Team ratings** – each team's attack and defence rating.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Host it

**Streamlit Community Cloud (free):** push this folder to a GitHub repository,
sign in at share.streamlit.io, choose "Create app", pick the repository and set
the main file to `app.py`.

**Any other host (Render, Railway, Fly.io, a VPS):** install the requirements
and start it with

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

## Leagues

| League | Reported chances |
|---|---|
| Premier League | title, top N (4 to 7), relegation (bottom 3) |
| Championship | title, automatic promotion (top 2), play-offs (3rd to 8th), relegation (bottom 3) |
| League One | title, automatic promotion (top 2), play-offs (3rd to 6th), relegation (bottom 4) |
| League Two | title, automatic promotion (top 3), play-offs (4th to 7th), relegation (bottom 2) |

These are set in `LEAGUES` at the top of `app.py`; change the numbers there if
the rules change. Play-off winners are not simulated: the app gives the chance
of finishing in the play-off places.

Only the Premier League has bundled results. The other three leagues are
downloaded from football-data.co.uk when selected, so they need that download
to work (or a CSV uploaded in the sidebar).

Teams new to a division start below average if promoted and above average if
they came down from the division above.

`data/adjustments.csv` holds points deductions (League, Team, Points, Note).
They are applied to the current table and the simulation. Team names must match
the results data.

## Data

- `data/results.csv` – every 2025-26 result and the 2026-27 results up to
  8 November 2026 (10 games each). Columns: Season, Date, HomeTeam, AwayTeam,
  FTHG, FTAG. Team names follow football-data.co.uk ("Man City", "Nott'm Forest").
- `data/fixtures.csv` – dates for upcoming fixtures (currently the
  21-23 November round). Optional: the simulation does not need dates, because
  the remaining fixtures are worked out as every home/away pairing without a
  result yet. Add rows here to show more dated rounds on the Fixtures tab.

Keeping results current, in order of preference:

1. **Automatic** – with "Fetch latest results automatically" on, the app
   downloads last season and this season from football-data.co.uk every six
   hours. If the download fails it uses `data/results.csv` and says so.
2. **Upload** – upload a CSV in the sidebar (a football-data.co.uk file works
   as is).
3. **Edit the file** – append rows to `data/results.csv` and redeploy.

At the start of a new season the app needs no code changes: it reads the teams
from the data and treats clubs with no earlier top-flight matches as promoted.

## The model

Goals are Poisson. Each team has an attack and a defence rating, plus a shared
home advantage:

    home goals rate = exp(base + home advantage + attack[home] - defence[away])
    away goals rate = exp(base + attack[away] - defence[home])

- Older matches count less (default half-life 365 days).
- Ratings are pulled towards league average, which stops ten games of form
  being over-read. Newly promoted teams start below average.
- A Dixon-Coles adjustment corrects the 0-0, 1-0, 0-1 and 1-1 scores, which a
  plain Poisson model gets slightly wrong.
- The season simulation plays each remaining fixture, adds the points to the
  real table and ranks by points, goal difference, then goals scored. By
  default each simulated season also draws slightly different team ratings, so
  the projections reflect how uncertain the ratings are this early on.

All three settings are sliders in the sidebar. The model is in `model.py` and
has no Streamlit dependency, so it can be used from a notebook or script.

## How accurate is it?

`python backtest.py` re-runs a walk-forward test: for each match day the model
is fitted only on earlier matches. On the 387 matches from 1 November 2025 to
8 November 2026 it picked the right result 46% of the time with a log loss of
1.04, against 40% and 1.10 for always predicting the historical
home/draw/away rates. That is a solid results-only model; it is not expected
to beat bookmaker odds, and it knows nothing about injuries, transfers or
managers.
