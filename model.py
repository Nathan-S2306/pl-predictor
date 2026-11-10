"""Goals model and league simulator.

Model
-----
Home and away goals are Poisson with rates

    log(home rate) = intercept + home_adv + attack[home] - defence[away]
    log(away rate) = intercept            + attack[away] - defence[home]

fitted by penalised maximum likelihood:

* older matches are down-weighted (half-life in days),
* attack/defence ratings are shrunk towards a prior (0 = league average;
  newly promoted teams get a below-average prior),
* a Dixon-Coles correction (rho) adjusts the 0-0, 1-0, 0-1 and 1-1 scores.

The fit also returns the approximate covariance of the ratings, so the season
simulation can draw a slightly different set of ratings in every run instead
of treating the estimates as exact.

Nothing in here depends on Streamlit; it only needs numpy, pandas and scipy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

MAX_GOALS = 10  # scoreline grid is 0..MAX_GOALS for each side


# ---------------------------------------------------------------- data ----

def season_of(date: pd.Timestamp) -> str:
    """Season label for a date, e.g. 2026-11-08 -> '2026-27' (season starts in July)."""
    start = date.year if date.month >= 7 else date.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


def clean_results(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a results table to Season, Date, HomeTeam, AwayTeam, FTHG, FTAG.

    Accepts football-data.co.uk style columns. Rows without a final score
    (fixtures not yet played) are dropped.
    """
    df = df.rename(columns={c: c.strip() for c in df.columns})
    needed = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Results file is missing columns: {', '.join(missing)}")
    out = df[needed].copy()
    raw = out["Date"].astype(str).str.strip()
    iso = raw.str.match(r"^\d{4}-\d{2}-\d{2}")
    dates = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
    if iso.any():
        dates[iso] = pd.to_datetime(raw[iso].str[:10], format="%Y-%m-%d", errors="coerce")
    if (~iso).any():  # football-data.co.uk uses dd/mm/yy or dd/mm/yyyy
        dates[~iso] = pd.to_datetime(raw[~iso], dayfirst=True, format="mixed", errors="coerce")
    out["Date"] = dates
    out["FTHG"] = pd.to_numeric(out["FTHG"], errors="coerce")
    out["FTAG"] = pd.to_numeric(out["FTAG"], errors="coerce")
    out = out.dropna(subset=needed)
    out["HomeTeam"] = out["HomeTeam"].astype(str).str.strip()
    out["AwayTeam"] = out["AwayTeam"].astype(str).str.strip()
    out[["FTHG", "FTAG"]] = out[["FTHG", "FTAG"]].astype(int)
    if "Season" in df.columns:
        out["Season"] = df.loc[out.index, "Season"].astype(str)
    else:
        out["Season"] = out["Date"].map(season_of)
    out = out.sort_values(["Date", "HomeTeam"]).reset_index(drop=True)
    return out[["Season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]]


def current_season(results: pd.DataFrame) -> str:
    return results.loc[results["Date"].idxmax(), "Season"]


def league_table(results: pd.DataFrame, teams: list[str] | None = None) -> pd.DataFrame:
    """Standard league table (3 points a win; sorted by points, GD, goals for)."""
    teams = teams or sorted(set(results["HomeTeam"]) | set(results["AwayTeam"]))
    idx = {t: i for i, t in enumerate(teams)}
    tab = np.zeros((len(teams), 6), dtype=int)  # P W D L GF GA
    for h, a, hg, ag in results[["HomeTeam", "AwayTeam", "FTHG", "FTAG"]].itertuples(index=False):
        for t, gf, ga in ((h, hg, ag), (a, ag, hg)):
            r = tab[idx[t]]
            r[0] += 1
            r[1 if gf > ga else 2 if gf == ga else 3] += 1
            r[4] += gf
            r[5] += ga
    out = pd.DataFrame(tab, index=teams, columns=["P", "W", "D", "L", "GF", "GA"])
    out["GD"] = out["GF"] - out["GA"]
    out["Pts"] = 3 * out["W"] + out["D"]
    out = out.sort_values(["Pts", "GD", "GF"], ascending=False, kind="stable")
    out.insert(0, "Pos", range(1, len(out) + 1))
    out.index.name = "Team"
    return out


def remaining_fixtures(results: pd.DataFrame, schedule: pd.DataFrame | None = None) -> pd.DataFrame:
    """Unplayed fixtures of the current season.

    Every team plays every other team home and away, so the remaining fixtures
    are all home/away pairings that do not yet have a result. Dates are taken
    from `schedule` (Date, HomeTeam, AwayTeam) where available.
    """
    season = current_season(results)
    cur = results[results["Season"] == season]
    teams = sorted(set(cur["HomeTeam"]) | set(cur["AwayTeam"]))
    played = set(zip(cur["HomeTeam"], cur["AwayTeam"]))
    rows = [(h, a) for h in teams for a in teams if h != a and (h, a) not in played]
    fx = pd.DataFrame(rows, columns=["HomeTeam", "AwayTeam"])
    fx["Date"] = pd.NaT
    if schedule is not None and len(schedule):
        s = schedule.copy()
        s["Date"] = pd.to_datetime(s["Date"], errors="coerce")
        s = s.dropna(subset=["Date"]).drop_duplicates(["HomeTeam", "AwayTeam"], keep="last")
        fx = fx.drop(columns="Date").merge(s[["Date", "HomeTeam", "AwayTeam"]], how="left", on=["HomeTeam", "AwayTeam"])
    fx = fx.sort_values(["Date", "HomeTeam"], na_position="last").reset_index(drop=True)
    return fx[["Date", "HomeTeam", "AwayTeam"]]


# --------------------------------------------------------------- model ----

@dataclass
class GoalsModel:
    teams: list[str]
    intercept: float
    home_adv: float
    attack: np.ndarray
    defence: np.ndarray
    rho: float
    cov: np.ndarray  # covariance of [intercept, home_adv, attack..., defence...]
    fitted_on: int = 0
    last_date: pd.Timestamp | None = None
    promoted: list[str] = field(default_factory=list)

    def __post_init__(self):
        self._idx = {t: i for i, t in enumerate(self.teams)}

    # -- single match ------------------------------------------------------
    def rates(self, home: str, away: str, neutral: bool = False) -> tuple[float, float]:
        i, j = self._idx[home], self._idx[away]
        h = 0.0 if neutral else self.home_adv
        lam = np.exp(self.intercept + h + self.attack[i] - self.defence[j])
        mu = np.exp(self.intercept + self.attack[j] - self.defence[i])
        return float(lam), float(mu)

    def score_matrix(self, home: str, away: str, neutral: bool = False) -> np.ndarray:
        """P[i, j] = probability the match ends home i - j away."""
        lam, mu = self.rates(home, away, neutral)
        return score_matrix(lam, mu, self.rho)

    def predict(self, home: str, away: str, neutral: bool = False) -> dict:
        lam, mu = self.rates(home, away, neutral)
        P = score_matrix(lam, mu, self.rho)
        g = np.arange(P.shape[0])
        i, j = np.unravel_index(np.argmax(P), P.shape)
        total = g[:, None] + g[None, :]
        p_home, p_draw, p_away = float(np.tril(P, -1).sum()), float(np.trace(P)), float(np.triu(P, 1).sum())
        # Predicted score: the most likely score *within the most likely result*, so a favourite is
        # shown winning rather than every close game reading 1-1.
        diff = g[:, None] - g[None, :]
        mask = diff > 0 if p_home >= max(p_draw, p_away) else diff < 0 if p_away >= p_draw else diff == 0
        ri, rj = np.unravel_index(np.argmax(np.where(mask, P, -1.0)), P.shape)
        return {
            "result_score": (int(ri), int(rj)),
            "home": home, "away": away,
            "home_xg": lam, "away_xg": mu,
            "p_home": float(np.tril(P, -1).sum()),
            "p_draw": float(np.trace(P)),
            "p_away": float(np.triu(P, 1).sum()),
            "likely_score": (int(i), int(j)),
            "likely_score_p": float(P[i, j]),
            "p_over_2_5": float(P[total >= 3].sum()),
            "p_btts": float(P[1:, 1:].sum()),
            "matrix": P,
        }

    def ratings(self) -> pd.DataFrame:
        """Team ratings as goals multipliers relative to an average team (1.00)."""
        se = np.sqrt(np.clip(np.diag(self.cov), 0, None))
        n = len(self.teams)
        df = pd.DataFrame({
            "Team": self.teams,
            "Attack": np.exp(self.attack),
            "Defence": np.exp(-self.defence),  # goals conceded multiplier; lower is better
            "attack_log": self.attack,
            "defence_log": self.defence,
            "attack_se": se[2:2 + n],
            "defence_se": se[2 + n:2 + 2 * n],
        })
        df["Overall"] = df["attack_log"] + df["defence_log"]
        return df.sort_values("Overall", ascending=False).reset_index(drop=True)


def _dc_tau(x, y, lam, mu, rho):
    """Dixon-Coles low-score adjustment factor."""
    tau = np.ones(np.broadcast(x, y, lam, mu).shape)
    x, y, lam, mu = np.broadcast_arrays(x, y, lam, mu)
    m = (x == 0) & (y == 0); tau[m] = 1 - lam[m] * mu[m] * rho
    m = (x == 0) & (y == 1); tau[m] = 1 + lam[m] * rho
    m = (x == 1) & (y == 0); tau[m] = 1 + mu[m] * rho
    m = (x == 1) & (y == 1); tau[m] = 1 - rho
    return tau


def score_matrix(lam: float, mu: float, rho: float = 0.0, max_goals: int = MAX_GOALS) -> np.ndarray:
    g = np.arange(max_goals + 1)
    P = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
    P[:2, :2] *= np.array([[1 - lam * mu * rho, 1 + lam * rho], [1 + mu * rho, 1 - rho]]).clip(min=1e-9)
    return P / P.sum()


def fit(results: pd.DataFrame, half_life_days: float = 365.0, prior_sd: float = 0.30,
        promoted_offset: float = 0.20, as_of: pd.Timestamp | None = None) -> GoalsModel:
    """Fit the goals model.

    half_life_days   a match this many days old counts half as much as one today
    prior_sd         prior spread of attack/defence ratings (log scale); smaller = more shrinkage
    promoted_offset  teams with no matches before the current season start this far
                     below average in both attack and defence (log scale) until results say otherwise
    """
    res = results.sort_values("Date")
    as_of = pd.Timestamp(as_of) if as_of is not None else res["Date"].max()
    res = res[res["Date"] <= as_of]
    if len(res) < 20:
        raise ValueError("Need at least 20 played matches to fit the model.")
    season = current_season(res)
    cur = res[res["Season"] == season]
    teams = sorted(set(res["HomeTeam"]) | set(res["AwayTeam"]))
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}
    hi = res["HomeTeam"].map(idx).to_numpy()
    ai = res["AwayTeam"].map(idx).to_numpy()
    hg = res["FTHG"].to_numpy(float)
    ag = res["FTAG"].to_numpy(float)
    age = (as_of - res["Date"]).dt.days.to_numpy(float)
    w = 0.5 ** (age / half_life_days)
    m = len(res)

    # Design matrix: one row per team per match. Columns: intercept, home, attack(n), defence(n).
    p = 2 + 2 * n
    X = np.zeros((2 * m, p))
    r = np.arange(m)
    X[:m, 0] = 1; X[:m, 1] = 1; X[r, 2 + hi] = 1; X[r, 2 + n + ai] = -1          # home goals
    X[m:, 0] = 1;               X[m + r, 2 + ai] = 1; X[m + r, 2 + n + hi] = -1  # away goals
    y = np.concatenate([hg, ag])
    ww = np.concatenate([w, w])

    # Prior: ratings ~ Normal(mean, prior_sd). Promoted teams (no earlier-season matches) start below average.
    earlier = res[res["Season"] != season]
    old_teams = set(earlier["HomeTeam"]) | set(earlier["AwayTeam"])
    promoted = sorted(set(cur["HomeTeam"]) | set(cur["AwayTeam"])) if len(earlier) else []
    promoted = [t for t in promoted if t not in old_teams]
    prior_mean = np.zeros(p)
    for t in promoted:
        prior_mean[2 + idx[t]] = -promoted_offset
        prior_mean[2 + n + idx[t]] = -promoted_offset
    pen = np.zeros(p)
    pen[2:] = 1.0 / prior_sd ** 2

    def nll(beta):
        eta = X @ beta
        mu_ = np.exp(eta)
        d = beta - prior_mean
        val = np.sum(ww * (mu_ - y * eta)) + 0.5 * np.sum(pen * d * d)
        grad = X.T @ (ww * (mu_ - y)) + pen * d
        return val, grad

    beta0 = np.zeros(p)
    beta0[0] = np.log(max(y.mean(), 0.1))
    opt = minimize(nll, beta0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
    beta = opt.x
    mu_hat = np.exp(X @ beta)
    H = X.T @ (X * (ww * mu_hat)[:, None]) + np.diag(pen)
    cov = np.linalg.inv(H)

    # Dixon-Coles rho, estimated with the rates held fixed.
    lam, mu = mu_hat[:m], mu_hat[m:]
    low = (hg <= 1) & (ag <= 1)

    def neg_ll_rho(rho):
        tau = _dc_tau(hg[low], ag[low], lam[low], mu[low], rho)
        return -np.sum(w[low] * np.log(np.clip(tau, 1e-9, None)))

    rho = float(minimize_scalar(neg_ll_rho, bounds=(-0.2, 0.2), method="bounded").x) if low.any() else 0.0

    return GoalsModel(teams=teams, intercept=float(beta[0]), home_adv=float(beta[1]),
                      attack=beta[2:2 + n].copy(), defence=beta[2 + n:].copy(), rho=rho, cov=cov,
                      fitted_on=m, last_date=as_of, promoted=promoted)


# ---------------------------------------------------------- simulation ----

def sample_scores(model: GoalsModel, fixtures: pd.DataFrame, seed: int | None = None) -> list[tuple[int, int]]:
    """Draw one random score for each fixture from the model's scoreline probabilities.

    This is a single simulated run: scores vary the way real ones do, but any
    one run is just one of many possible outcomes.
    """
    rng = np.random.default_rng(seed)
    out = []
    for h, a in fixtures[["HomeTeam", "AwayTeam"]].itertuples(index=False):
        P = model.score_matrix(h, a)
        k = rng.choice(P.size, p=P.ravel())
        out.append((int(k // P.shape[1]), int(k % P.shape[1])))
    return out


@dataclass
class SimulationResult:
    teams: list[str]
    position_probs: pd.DataFrame   # team x finishing position (1..n), probabilities
    summary: pd.DataFrame          # one row per team
    points: np.ndarray             # n_sims x n_teams final points
    n_sims: int


def simulate_season(model: GoalsModel, results: pd.DataFrame, fixtures: pd.DataFrame | None = None,
                    n_sims: int = 10_000, seed: int | None = 42, rating_uncertainty: bool = True,
                    top_n: int = 4, relegated: int = 3, chunk: int = 2_000) -> SimulationResult:
    """Monte Carlo simulation of the rest of the current season.

    Each run plays every remaining fixture once, adds the points to the real
    table so far, and ranks teams by points, goal difference, then goals scored
    (remaining ties broken at random).
    """
    season = current_season(results)
    cur = results[results["Season"] == season]
    teams = sorted(set(cur["HomeTeam"]) | set(cur["AwayTeam"]))
    n = len(teams)
    if fixtures is None:
        fixtures = remaining_fixtures(results)
    now = league_table(cur, teams).loc[teams]
    base_pts = now["Pts"].to_numpy(float)
    base_gd = now["GD"].to_numpy(float)
    base_gf = now["GF"].to_numpy(float)

    rng = np.random.default_rng(seed)
    tix = {t: i for i, t in enumerate(teams)}
    mix = model._idx
    nm = len(model.teams)
    f = len(fixtures)
    h = fixtures["HomeTeam"].map(tix).to_numpy()
    a = fixtures["AwayTeam"].map(tix).to_numpy()
    mh = fixtures["HomeTeam"].map(mix).to_numpy()
    ma = fixtures["AwayTeam"].map(mix).to_numpy()

    # Fixture design rows in the model's parameter layout.
    p = 2 + 2 * nm
    Xh = np.zeros((f, p)); Xa = np.zeros((f, p))
    r = np.arange(f)
    Xh[:, 0] = 1; Xh[:, 1] = 1; Xh[r, 2 + mh] = 1; Xh[r, 2 + nm + ma] = -1
    Xa[:, 0] = 1;               Xa[r, 2 + ma] = 1; Xa[r, 2 + nm + mh] = -1
    Hm = np.zeros((f, n)); Hm[r, h] = 1   # fixture -> home team
    Am = np.zeros((f, n)); Am[r, a] = 1   # fixture -> away team
    beta = np.concatenate([[model.intercept, model.home_adv], model.attack, model.defence])
    L = np.linalg.cholesky(model.cov + 1e-12 * np.eye(p)) if rating_uncertainty else None
    rho = model.rho

    pos_counts = np.zeros((n, n))
    gf_sum = np.zeros(n)
    ga_sum = np.zeros(n)
    all_pts = np.empty((n_sims, n), dtype=np.int16)
    done = 0
    while done < n_sims:
        k = min(chunk, n_sims - done)
        if f:
            B = beta + rng.standard_normal((k, p)) @ L.T if rating_uncertainty else np.broadcast_to(beta, (k, p))
            lam = np.exp(B @ Xh.T)
            mu = np.exp(B @ Xa.T)
            hg = rng.poisson(lam)
            ag = rng.poisson(mu)
            if rho != 0.0:  # Dixon-Coles: reshuffle probability between 0-0, 0-1, 1-0, 1-1
                low = (hg <= 1) & (ag <= 1)
                l_, m_ = lam[low], mu[low]
                w00 = np.clip(1 - l_ * m_ * rho, 1e-9, None)
                w01 = m_ * np.clip(1 + l_ * rho, 1e-9, None)
                w10 = l_ * np.clip(1 + m_ * rho, 1e-9, None)
                w11 = l_ * m_ * np.clip(1 - rho, 1e-9, None)
                tot = w00 + w01 + w10 + w11
                u = rng.random(l_.shape) * tot
                cell = (u > w00).astype(int) + (u > w00 + w01) + (u > w00 + w01 + w10)
                hg[low] = cell >= 2
                ag[low] = (cell == 1) | (cell == 3)
            hp = np.where(hg > ag, 3.0, np.where(hg == ag, 1.0, 0.0))
            ap = np.where(ag > hg, 3.0, np.where(hg == ag, 1.0, 0.0))
            pts = base_pts + hp @ Hm + ap @ Am
            gf = base_gf + hg @ Hm + ag @ Am
            gd = base_gd + (hg - ag) @ Hm + (ag - hg) @ Am
        else:
            pts = np.tile(base_pts, (k, 1)); gf = np.tile(base_gf, (k, 1)); gd = np.tile(base_gd, (k, 1))
        key = pts * 1e7 + (gd + 1000) * 1e3 + gf + rng.random((k, n)) * 0.5
        order = np.argsort(-key, axis=1)             # team index at each position
        pos = np.empty_like(order)
        np.put_along_axis(pos, order, np.arange(n)[None, :].repeat(k, 0), axis=1)  # position of each team
        for t in range(n):
            pos_counts[t] += np.bincount(pos[:, t], minlength=n)
        all_pts[done:done + k] = pts
        gf_sum += gf.sum(0)
        ga_sum += (gf - gd).sum(0)
        done += k

    probs = pos_counts / n_sims
    position_probs = pd.DataFrame(probs, index=teams, columns=range(1, n + 1))
    summary = pd.DataFrame({
        "Team": teams,
        "Played": now["P"].to_numpy(),
        "Pts now": base_pts.astype(int),
        "Exp pts": all_pts.mean(0),
        "Pts low": np.percentile(all_pts, 5, axis=0),
        "Pts high": np.percentile(all_pts, 95, axis=0),
        "Exp GF": gf_sum / n_sims,
        "Exp GA": ga_sum / n_sims,
        "Avg pos": probs @ np.arange(1, n + 1),
        "Title": probs[:, 0],
        f"Top {top_n}": probs[:, :top_n].sum(1),
        "Relegation": probs[:, n - relegated:].sum(1) if relegated else np.zeros(n),
    }).sort_values(["Exp pts", "Title"], ascending=False).reset_index(drop=True)
    summary.insert(0, "Pos", range(1, n + 1))
    position_probs = position_probs.loc[summary["Team"]]
    return SimulationResult(teams=teams, position_probs=position_probs, summary=summary,
                            points=all_pts, n_sims=n_sims)


# ------------------------------------------------------------ backtest ----

def backtest(results: pd.DataFrame, start: str | pd.Timestamp, **fit_kwargs) -> pd.DataFrame:
    """Walk-forward test: for each match day from `start`, fit on earlier matches only and predict.

    Returns one row per predicted match with the model's home/draw/away
    probabilities, so accuracy and log loss can be compared with a baseline.
    """
    res = results.sort_values("Date")
    rows = []
    for day in sorted(res.loc[res["Date"] >= pd.Timestamp(start), "Date"].unique()):
        train = res[res["Date"] < day]
        today = res[res["Date"] == day]
        try:
            m = fit(train, as_of=pd.Timestamp(day) - pd.Timedelta(days=1), **fit_kwargs)
        except ValueError:
            continue
        for r in today.itertuples():
            if r.HomeTeam not in m._idx or r.AwayTeam not in m._idx:
                continue
            pr = m.predict(r.HomeTeam, r.AwayTeam)
            outcome = "H" if r.FTHG > r.FTAG else "D" if r.FTHG == r.FTAG else "A"
            rows.append({"Date": r.Date, "HomeTeam": r.HomeTeam, "AwayTeam": r.AwayTeam,
                         "FTHG": r.FTHG, "FTAG": r.FTAG, "outcome": outcome,
                         "p_home": pr["p_home"], "p_draw": pr["p_draw"], "p_away": pr["p_away"],
                         "home_xg": pr["home_xg"], "away_xg": pr["away_xg"]})
    return pd.DataFrame(rows)
