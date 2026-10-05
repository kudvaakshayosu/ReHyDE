"""Constrained active-learning campaign on the chemostat.

Arms (all at the same budget). ReHyDE = Region-constrained Hybrid Dynamics
Exploration; each ablation removes exactly one of its named components:

  policy     model      name                      Re   Hy   E
  hybrid     hybrid     ReHyDE                    y    y    y
  blackbox   blackbox   ReHyDE w/o hybrid         y    -    y
  crandom    hybrid     ReHyDE w/o exploration    y    y    -
  random     hybrid     Random                    -    y    -

crandom draws uniformly but keeps only recipes satisfying the same chance
constraint, so it isolates the information objective. random is an
unconstrained uniform design and is evaluated with both models.
Metrics each round (on a fixed 40-recipe test set, noiseless plant):
  mae_X/mae_S -- MAE (g/L) of the mean rollout vs the plant, on the test
                 recipes that are feasible on the plant
  nmae_all   -- normalised MAE of the mean rollout vs truth, all test recipes
  nmae_feas  -- same, restricted to recipes that are truly feasible (S >= S_min)
  nmae_bnd   -- same, on the recipes nearest the feasibility boundary
                (true min S in [0.25, 1.5]) -- where characterisation matters
  feas_acc   -- accuracy of the model's feasibility call P(S>=S_min) >= 0.9
plus plant-side constraint violations of the experiments actually run.

Every trained model (every arm, repeat and round) is saved to
<out>/checkpoints/{policy}_s{seed}_r{round:02d}_{model}.pt; rebuild one with
models.load_model(path). Round r is the model trained on 3 + r experiments,
i.e. the one that chose experiment 4 + r.
"""
import os as _os, torch as _torch
# The tensors here are tiny (<= 65 inducing points, 5-unit layers), so thread
# synchronisation costs far more than the arithmetic: 2 threads measured 7x
# slower than 1 per rollout. Override with e.g.  TORCH_THREADS=4 python run.py
_torch.set_num_threads(int(_os.environ.get("TORCH_THREADS", "1")))

import argparse, json, os, time
import numpy as np
from scipy.stats.qmc import Sobol, LatinHypercube
from plant import simulate, from_unit, S_MIN
from models import Hybrid, BlackBox, save_model, ACQ_SIGMA
from acquisition import select, select_constrained_random, score, make_eps, ALPHA

N_INIT, N_ROUNDS, N_TEST = 3, 5, 64      # 3 initial + 5 designed = 8 experiments
REFIT_EVERY = 1          # cold DKL refit every k rounds, warm fine-tune otherwise (as CHO)
MODELS = {"hybrid": Hybrid, "blackbox": BlackBox}


def test_set():
    R = from_unit(Sobol(7, scramble=True, seed=12345).random_base2(6))
    Y = np.array([simulate(r, noise=False) for r in R])
    return R, Y


def evaluate(model, R, Ytrue, rng):
    eps = make_eps(model, 12, rng)
    _, p, traj = score(model, R, eps)
    pred = traj[:, 0]
    scale = np.abs(Ytrue).mean((0, 1))
    feas_true = (Ytrue[:, :, 1] >= S_MIN).all(1)
    err = (np.abs(pred - Ytrue) / scale).mean((1, 2))           # per recipe
    feas_pred = p >= 1 - ALPHA
    smin = Ytrue[:, :, 1].min(1)
    bnd = (smin >= 0.25) & (smin <= 1.5)
    # MAE between model and plant, over the test recipes that are FEASIBLE on the plant
    mae = np.abs(pred - Ytrue).mean(1)[feas_true]                # (n_feasible, 2)
    return {"mae_X": float(mae[:, 0].mean()), "mae_S": float(mae[:, 1].mean()),
            "nmae_all": float(err.mean()),
            "nmae_feas": float(err[feas_true].mean()) if feas_true.any() else np.nan,
            "nmae_bnd": float(err[bnd].mean()),
            "feas_acc": float((feas_pred == feas_true).mean())}


def campaign(policy, seed, Rt, Yt, log=print, ckpt_dir=None):
    rng = np.random.default_rng(seed)
    init = from_unit(LatinHypercube(7, seed=seed).random(N_INIT))
    data = [(r, simulate(r, rng=rng)) for r in init]
    kind = policy
    rows = []
    models = {}
    for rnd in range(N_ROUNDS + 1):
        need = list(MODELS) if kind == "random" else (
            ["hybrid"] if kind == "crandom" else [kind])
        cold = rnd % REFIT_EVERY == 0
        models = {k: MODELS[k]().fit(data, seed=seed, warm=None if cold else models[k])
                  for k in need}
        row = {"round": rnd, "n": len(data), "cold_fit": bool(cold),
               "acq_sigma": ACQ_SIGMA}
        if ckpt_dir:
            for k, m in models.items():
                save_model(m, os.path.join(ckpt_dir, f"{policy}_s{seed}_r{rnd:02d}_{k}.pt"),
                           policy=policy, seed=seed, round=rnd, n_experiments=len(data),
                           cold_fit=bool(cold))
        for k, m in models.items():
            row.update({f"{k}_{a}": v for a, v in evaluate(m, Rt, Yt, rng).items()})
        viol = [float(not (y[:, 1] >= S_MIN).all()) for _, y in data[N_INIT:]]
        row["violations"] = int(sum(viol))
        if rnd == N_ROUNDS:
            row["data"] = [[list(map(float, rr)), y.tolist()] for rr, y in data]
            rows.append(row); break
        if kind == "random":                       # unconstrained uniform design
            r = from_unit(rng.random(7)); a, p = np.nan, np.nan
        elif kind == "crandom":                    # uniform design, region-constrained
            r, a, p = select_constrained_random(models["hybrid"], rng)
        else:                                      # int ||sigma|| dt acquisition
            info = {}
            r, a, p = select(models[kind], rng, info=info)
            row.update({f"search_{k}": v for k, v in info.items()})
        y = simulate(r, rng=rng)
        row.update({"acq": float(a), "p_pred": float(p),
                    "S_min_plant": float(y[:, 1].min()),
                    "recipe": [float(v) for v in r]})
        data.append((r, y))
        rows.append(row)
        log(f"  {policy:14s} s{seed} r{rnd:2d} " +
            " ".join(f"{k}={v:.3f}" for k, v in row.items()
                     if isinstance(v, float) and k != "recipe"))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(15)))
    ap.add_argument("--policies", nargs="+",
                    default=["hybrid", "blackbox", "crandom", "random"])
    ap.add_argument("--out", default="results_final")
    ap.add_argument("--rounds", type=int, default=None, help="override N_ROUNDS (smoke tests)")
    a = ap.parse_args()
    if a.rounds is not None:
        N_ROUNDS = a.rounds
    os.makedirs(a.out, exist_ok=True)
    ckpt = os.path.join(a.out, "checkpoints"); os.makedirs(ckpt, exist_ok=True)
    print(f"acquisition integrates the '{ACQ_SIGMA}' sigma (models.ACQ_SIGMA)")
    Rt, Yt = test_set()
    sm = Yt[:, :, 1].min(1)
    print(f"test set: {int((sm >= S_MIN).sum())}/{N_TEST} truly feasible, "
          f"{int(((sm >= 0.25) & (sm <= 1.5)).sum())} near the boundary")
    for pol in a.policies:
        for s in a.seeds:
            t0 = time.time()
            rows = campaign(pol, s, Rt, Yt, ckpt_dir=ckpt)
            json.dump(rows, open(f"{a.out}/{pol}_s{s}.json", "w"), indent=1)
            print(f"{pol} seed {s}: {time.time()-t0:.0f}s")