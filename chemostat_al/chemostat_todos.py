"""Numbers for the chemostat TODOs in the paper.

  1. Feasible fraction of U: share of recipes whose noise-free plant
     trajectory satisfies S(t_k) >= S_min at every sampling time.
  2. Closure MAE / NLL inside (S >= S_min) and outside (S < S_min) the
     admissible region, at the final budget, for every hybrid-closure method.

usage: python chemostat_todos.py [results_dir] [n_recipes_for_fraction]
"""
import glob, json, os, sys
import numpy as np, torch
from scipy import stats
from scipy.stats.qmc import Sobol
torch.set_num_threads(1)
from plant import simulate, from_unit, mu_true, S_MIN
from models import load_model

d = sys.argv[1] if len(sys.argv) > 1 else "results_final"
N_FRAC = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
N_INIT = 3

# ---- 1. feasible fraction of U ------------------------------------------------
U = Sobol(7, scramble=True, seed=0).random(N_FRAC)
ok = np.array([simulate(from_unit(u), noise=False)[:, 1].min() >= S_MIN for u in U])
p = ok.mean(); half = 1.96 * np.sqrt(p * (1 - p) / N_FRAC)
print(f"1. Feasible fraction of U: {100*p:.1f}% +/- {100*half:.1f}% "
      f"({ok.sum()} of {N_FRAC} Sobol recipes, noise-free plant)\n")

# ---- 2. closure error inside / outside the admissible region -------------------
X, S = np.meshgrid(np.linspace(0, 6, 160), np.linspace(0, 10, 160))
X, S = X.ravel(), S.ravel()
g = mu_true(X, S)
inside = S >= S_MIN

def scores(model):
    m, s = model.mu(X, S)
    e, v = g - m, np.maximum(s, 1e-6) ** 2
    nll = 0.5 * np.log(2 * np.pi * v) + e ** 2 / (2 * v)
    return [np.abs(e[k]).mean() for k in (inside, ~inside)] + \
           [nll[k].mean() for k in (inside, ~inside)]

arms = [("hybrid", "ReHyDE"), ("crandom", "ReHyDE w/o exploration"), ("random", "Random")]
res = {}
for pol, name in arms:
    rows = []
    for f in sorted(glob.glob(f"{d}/{pol}_s*.json")):
        seed = int(os.path.basename(f).rsplit("_s", 1)[1].split(".")[0])
        last = len(json.load(open(f))) - 1                      # final round
        ck = f"{d}/checkpoints/{pol}_s{seed}_r{last:02d}_hybrid.pt"
        rows.append((seed, scores(load_model(ck)[0])))
    rows.sort(); res[name] = (np.array([r[0] for r in rows]), np.array([r[1] for r in rows]))

t = lambda n: stats.t.ppf(0.975, n - 1)
cols = ["MAE inside", "MAE outside", "NLL inside", "NLL outside"]
print(f"2. Closure metrics at the final budget, 160x160 grid over X in [0,6], S in [0,10]")
print(f"   inside: S >= {S_MIN} ({inside.mean()*100:.0f}% of grid), outside: S < {S_MIN}")
print(f"   {'method':24s}" + "".join(f"{c:>20s}" for c in cols))
for name, (_, A) in res.items():
    n = len(A)
    print(f"   {name:24s}" + "".join(
        f"{A[:, j].mean():12.4f} ± {t(n)*A[:, j].std(ddof=1)/np.sqrt(n):.4f}" for j in range(4)))
print("   (mean ± 95% CI over repeats; MAE in 1/h)")

print("\n   Paired Wilcoxon, ReHyDE vs each (two-sided, unadjusted):")
sR, R = res["ReHyDE"]
for name in ("ReHyDE w/o exploration", "Random"):
    sO, O = res[name]
    assert (sR == sO).all(), "repeats do not match"
    for j, c in enumerate(cols):
        dlt = R[:, j] - O[:, j]
        print(f"     vs {name:24s}{c:12s} ReHyDE lower in {int((dlt < 0).sum()):2d}/{len(dlt)}"
              f"   p = {stats.wilcoxon(dlt).pvalue:.4f}")
