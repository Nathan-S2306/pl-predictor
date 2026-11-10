"""Walk-forward accuracy check. Run: python backtest.py

For every match day from the start date, the model is fitted only on earlier
matches and asked to predict that day's games.
"""
import sys

import numpy as np
import pandas as pd

import model as M

start = sys.argv[1] if len(sys.argv) > 1 else "2025-11-01"
res = M.clean_results(pd.read_csv("data/results.csv"))
bt = M.backtest(res, start)
p = bt[["p_home", "p_draw", "p_away"]].to_numpy()
y = bt["outcome"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
base = []
for d in bt["Date"]:
    tr = res[res["Date"] < d]
    h, dr = (tr["FTHG"] > tr["FTAG"]).mean(), (tr["FTHG"] == tr["FTAG"]).mean()
    base.append((h, dr, 1 - h - dr))
base = np.array(base)
rows = np.arange(len(y))
print(f"Matches predicted: {len(y)} (from {start})")
print(f"Model    log loss {-np.log(p[rows, y]).mean():.3f}   correct result {np.mean(p.argmax(1) == y):.1%}")
print(f"Baseline log loss {-np.log(base[rows, y]).mean():.3f}   correct result {np.mean(base.argmax(1) == y):.1%}"
      "   (always predicts the historical home/draw/away rates)")
print(f"Goals per game: predicted {bt['home_xg'].mean():.2f}-{bt['away_xg'].mean():.2f}, "
      f"actual {bt['FTHG'].mean():.2f}-{bt['FTAG'].mean():.2f}")
