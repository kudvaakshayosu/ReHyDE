"""Publication figures from a results directory (v2).

Changes from figures.py
  * everything adapts to the campaign length found in the results (e.g. 3 + 5
    or 3 + 10 experiments): x-axis, cold-refit markers, kinetics snapshots
  * fig_kinetics redrawn on one linear (X, S) box: column 1 is the true
    mu(X, S); the other columns are the hybrid DKL's posterior mean (top) and
    standard deviation (bottom) after the first, middle and last number of
    experiments, on a fixed box X in [0, 6], S in [0, 10] g/L (BOX)
  * kinetics_metrics(): grid RMSE and Gaussian NLL of the learned mu for every
    repeat (figures show only repeat SEED, default 0)
  * fig_grid_metrics: average grid MAE and average grid NLL of the WITHHELD
    CLOSURE g(X, S) against the number of experiments, mean with a 95 %
    confidence band over repeats, on a uniform M_GRID x M_GRID grid over (X, S):

        MAE = (1/G) sum_p |g(x_p) - mu(x_p)|
        NLL = (1/G) sum_p [ 0.5 log(2 pi sigma^2(x_p))
                            + (g(x_p) - mu(x_p))^2 / (2 sigma^2(x_p)) ]

    Only the hybrid has a closure, so only its two arms appear; the black box
    learns the full right-hand side and never represents g. Cached in
    grid_metrics.json -- delete that file to recompute.


  fig_phase     plant phase portraits (X, S) of every experiment in one repeat,
                with the S >= S_min constraint                    (7.0 x 2.5 in)
  fig_mae       test-set MAE versus number of experiments         (7.0 x 2.3 in)
  fig_kinetics  true versus hybrid-learned mu after 3, 6, 9, 13 experiments
                                                                  (7.0 x 4.0 in)
Each is written as vector PDF (for the paper) and 300-dpi PNG. No figure titles:
captions belong in the manuscript. Colours: Okabe-Ito (colour-blind safe).

The kinetics figure uses the SAVED campaign models (results/checkpoints) when
present; otherwise it refits the hybrid cold on the first n experiments and
says so.

RMSE in the mean panels is over the feasible part of the box (S >= S_min).
Hatched band: S < S_min, never visited by the designed experiments.

usage: python figures_v2.py [results_dir] [repeat]
"""
import os as _os, torch as _torch
# The tensors here are tiny (<= 65 inducing points, 5-unit layers), so thread
# synchronisation costs far more than the arithmetic: 2 threads measured 7x
# slower than 1 per rollout. Override with e.g.  TORCH_THREADS=4 python run.py
_torch.set_num_threads(int(_os.environ.get("TORCH_THREADS", "1")))

import glob, json, os, sys
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import matplotlib.ticker as mticker
import torch
from plant import mu_true, Y, S_IN, S_MIN, D_at, T_END

d = sys.argv[1] if len(sys.argv) > 1 else "results_final"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 14     # repeat shown in the figures
N_INIT = 3
KINETICS_NS = (3, 5, 8)             # budgets shown in fig_kinetics / kinetics_metrics
BOX = ((0.0, 6.0), (0.0, 10.0))     # fixed (X, S) box [g/L] for the kinetics grid
N_GRID = 160                        # grid points per axis (kinetics figure)
KIN_SNAPSHOTS = (3, 5, 8)           # budgets shown in fig_kinetics and kinetics_metrics;
                                    # None = initial, midpoint and final budget
M_GRID = 80                         # grid points per axis for the closure metrics
FEASIBLE_ONLY = True                # metric grid restricted to the feasible zone S >= S_min
SD_FLOOR = 1e-6                     # floor on sigma, so the NLL cannot divide by 0
PLOT_FLOOR = 1e-4                   # floor applied for log-scale display only

# --- trajectory metrics ------------------------------------------------------
N_TRAJ = 100                        # feasible test trajectories
N_MC = 64                           # posterior realizations propagated per model
TRAJ_SEED = 20260927                # fixed: the test set must not move between runs
COV_LEVEL = 0.95                    # nominal coverage of the predictive interval
VAR_FLOOR = 1e-12

# Okabe-Ito (colour-blind safe) + a red reserved for the constraint
C_FULL, C_NOHY, C_NOEX, C_RAND, CON = "#0072B2", "#D55E00", "#009E73", "#7F7F7F", "#CC0000"
HYB, BB, RND = C_FULL, C_NOHY, C_RAND          # backwards-compatible aliases

# ReHyDE = Region-constrained Hybrid Dynamics Exploration. Each ablation removes
# exactly one named component.  (policy, model, label, colour, linestyle, marker)
A_FULL = ("hybrid",   "hybrid",   "ReHyDE",                 C_FULL, "-",  "o")
A_NOHY = ("blackbox", "blackbox", "ReHyDE w/o hybrid",      C_NOHY, "--", "s")
A_NOEX = ("crandom",  "hybrid",   "ReHyDE w/o exploration", C_NOEX, "-.", "^")
A_RAND = ("random",   "hybrid",   "Random",                 C_RAND, ":",  "D")
ALL_ARMS = [A_FULL, A_NOHY, A_NOEX, A_RAND]


def bottom_legend(fig, handles, labels, ncol=4, gap=0.012):
    """Every legend that sits outside the axes goes below the figure, a fixed
    small gap under the lowest drawn element (tick labels included), so the
    spacing is identical across figures whatever their layout."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    y0 = min(inv.transform(a.get_tightbbox(r))[0, 1] for a in fig.axes
             if a.get_visible() and a.axison)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y0 - gap),
               ncol=ncol, handlelength=2.2, columnspacing=1.4)
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "lines.linewidth": 1.2, "legend.frameon": False,
    "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def panel(ax, letter, x=0.0, y=1.03):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontweight="bold",
            fontsize=8.5, ha="left", va="bottom")


_FIG_NAMES = ("fig_phase", "fig_mae", "fig_kinetics", "fig_grid_metrics",
              "fig_trajectory")


def clean_outputs():
    """Delete previously generated figures in the output directory, so a run
    can never leave a stale or half-renamed (*_new) duplicate behind."""
    gone = 0
    for name in _FIG_NAMES:
        for ext in ("pdf", "png"):
            for p in (f"{d}/{name}.{ext}", f"{d}/{name}_new.{ext}"):
                if os.path.exists(p):
                    try:
                        os.remove(p); gone += 1
                    except OSError:
                        print(f"  could not delete {p} -- close it and rerun")
    if gone:
        print(f"  removed {gone} figure file(s) from a previous run")


def save(fig, name):
    """Write PDF + PNG. If a file is locked (open in a viewer, or held by a sync
    client such as OneDrive -- on Windows this raises OSError), write
    <name>_new.<ext> instead of crashing, and say so."""
    os.makedirs(d, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.abspath(os.path.join(d, f"{name}.{ext}"))
        try:
            fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
        except OSError as e:
            alt = os.path.abspath(os.path.join(d, f"{name}_new.{ext}"))
            print(f"  could not overwrite {path}\n    ({e})\n"
                  f"    is it open in a viewer or being synced? wrote {alt} instead")
            fig.savefig(alt, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def load(pol, seed):
    return json.load(open(f"{d}/{pol}_s{seed}.json"))


def dense(recipe):
    Dt = lambda t: D_at(recipe, min(t, T_END - 1e-9))
    f = lambda t, z: [mu_true(z[0], max(z[1], 0)) * z[0] - Dt(t) * z[0],
                      -mu_true(z[0], max(z[1], 0)) * z[0] / Y + Dt(t) * (S_IN - z[1])]
    t = np.linspace(0, T_END, 401)
    return solve_ivp(f, (0, T_END), recipe[:2], t_eval=t, max_step=0.05, rtol=1e-8).y


# ------------------------------------------------------------------ phase
def fig_phase():
    arms = [(a[0], a[2]) for a in ALL_ARMS
            if glob.glob(f"{d}/{a[0]}_s{SEED}.json")]
    n_tot = len(load(arms[0][0], SEED)[-1]["data"])
    norm = colors.Normalize(N_INIT + 1, n_tot); cmap = cm.viridis
    fig, ax = plt.subplots(1, len(arms), figsize=(7.0, 2.4), sharey=True,
                           gridspec_kw={"wspace": 0.08})
    ax = np.atleast_1d(ax)
    for i, (a, (pol, title)) in enumerate(zip(ax, arms)):
        data = load(pol, SEED)[-1]["data"]
        a.axhspan(-1, S_MIN, color=CON, alpha=0.08, lw=0)
        a.axhline(S_MIN, color=CON, lw=0.9)
        for j, (r, yobs) in enumerate(data):
            X, S = dense(np.array(r)); yobs = np.array(yobs)
            init = j < N_INIT
            c = "0.55" if init else cmap(norm(j + 1))
            a.plot(X, S, color=c, ls="--" if init else "-", lw=0.8 if init else 1.1,
                   zorder=1 if init else 2)
            a.plot(yobs[:, 0], yobs[:, 1], "o", ms=1.6, color="k", zorder=3)
            a.plot(X[0], S[0], "o", ms=3.5, mfc="white", mec=c, mew=0.9, zorder=4)
            if S.min() < S_MIN:
                k = np.argmin(S)
                a.plot(X[k], S[k], "x", color=CON, ms=6, mew=1.6, zorder=5)
        nv = sum(np.array(y)[:, 1].min() < S_MIN for _, y in data[N_INIT:])
        # quarter-width panels: break "ReHyDE w/o X" so it clears the panel letter
        a.set_title(title.replace("ReHyDE w/o ", "ReHyDE\nw/o "), loc="right",
                    fontsize=7, linespacing=1.1)
        a.text(0.97, 0.975, f"infeasible: {nv}/{len(data) - N_INIT}",
               transform=a.transAxes, ha="right", va="top", fontsize=6,
               bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0))
        a.set_xlim(0, 4.6); a.set_ylim(-0.3, 10.8)
        a.set_xlabel(r"Biomass $X$ (g L$^{-1}$)")
        panel(a, "abcd"[i])
    ax[0].set_ylabel(r"Substrate $S$ (g L$^{-1}$)")
    ax[0].text(0.15, S_MIN + 0.15, r"$S_{\min}$", color=CON, fontsize=7)
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax.tolist(),
                      orientation="horizontal", pad=0.22, fraction=0.05, aspect=45)
    cb.set_label("Experiment number"); cb.outline.set_linewidth(0.5)
    cb.set_ticks(range(N_INIT + 1, n_tot + 1, 2 if n_tot > 9 else 1))
    save(fig, "fig_phase")


# ------------------------------------------------------------------ MAE
def curve(pol, key):
    runs = [json.load(open(f)) for f in sorted(glob.glob(f"{d}/{pol}_s*.json"))]
    M = np.array([[r.get(key, np.nan) for r in rows] for rows in runs], float)
    return np.array([r["n"] for r in runs[0]]), np.nanmean(M, 0), np.nanstd(M, 0)


def cold_n():
    """Numbers of experiments at which the DKL was refitted cold (from the logs)."""
    rows = load("hybrid", SEED)
    if "cold_fit" in rows[0]:
        ns = [r["n"] for r in rows if r.get("cold_fit")]
    else:
        from run import REFIT_EVERY
        ns = [r["n"] for r in rows if r["round"] % REFIT_EVERY == 0]
    # refit cold every round: the markers would sit on every point and say nothing
    return [] if len(ns) == len(rows) else ns


def fig_mae():
    arms = ALL_ARMS
    panels = [("mae_X", r"MAE of $X$ (g L$^{-1}$)", 1),
              ("mae_S", r"MAE of $S$ (g L$^{-1}$)", 1),
              ("nmae_bnd", "NMAE near constraint (%)", 100)]
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.3), gridspec_kw={"wspace": 0.38})
    for i, (a, (key, lab, sc)) in enumerate(zip(ax, panels)):
        for pol, mdl, name, c, ls, mk in arms:
            n, m, s = curve(pol, f"{mdl}_{key}")
            a.fill_between(n, sc * (m - s), sc * (m + s), color=c,
                           alpha=0.15 if ls == "-" else 0.07, lw=0)
            a.plot(n, sc * m, ls, color=c, marker=mk, ms=2.8,
                   mfc=c if ls == "-" else "white", mew=0.8, label=name)
        for nc in cold_n():
            a.axvline(nc, color="0.8", lw=0.6, ls=":", zorder=0)
        a.set_yscale("log"); a.set_xlabel("Number of experiments"); a.set_ylabel(lab)
        a.yaxis.set_major_locator(mticker.LogLocator(subs=(1, 2, 5)))
        a.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        a.yaxis.set_minor_formatter(mticker.NullFormatter())
        n_max = int(n.max()); step = 1 if n_max <= 9 else 2
        a.set_xticks(range(N_INIT, n_max + 1, step)); a.set_xlim(N_INIT - 0.4, n_max + 0.4)
        panel(a, "abc"[i])
    h, l = ax[0].get_legend_handles_labels()
    if cold_n():
        h.append(plt.Line2D([], [], color="0.7", lw=0.6, ls=":")); l.append("cold DKL refit")
    bottom_legend(fig, h, l, ncol=5)
    save(fig, "fig_mae")


# ------------------------------------------------------------------ kinetics
def model_at(n, data, pol="hybrid", kind="hybrid", seed=None):
    """The campaign's model of type `kind` after n experiments of arm `pol`
    (hybrid / blackbox = that model's acquisition arm; random = random designs).
    Uses the saved checkpoint when present, else refits cold on the first n
    experiments of that campaign. Returns (model, "saved" | "refit")."""
    seed = SEED if seed is None else seed
    path = f"{d}/checkpoints/{pol}_s{seed}_r{n - N_INIT:02d}_{kind}.pt"
    if os.path.exists(path):
        from models import load_model
        return load_model(path)[0], "saved"
    import models as _m
    cls = _m.Hybrid if kind == "hybrid" else _m.BlackBox
    return cls().fit(data[:n], seed=seed), "refit"


def hybrid_at(n, data, pol="hybrid", seed=None):
    return model_at(n, data, pol, "hybrid", seed)


def kinetics_grid(box=BOX, n_grid=N_GRID):
    (x_lo, x_hi), (s_lo, s_hi) = box
    Xg, Sg = np.meshgrid(np.linspace(x_lo, x_hi, n_grid), np.linspace(s_lo, s_hi, n_grid))
    return Xg, Sg


def grid_scores(h, Xg, Sg, feasible_only=True):
    """Average grid MAE and NLL of the withheld closure on the KINETICS-FIGURE
    grid (N_GRID x N_GRID, feasible part by default), plus the mean and sd
    surfaces the figure draws. Same definition as closure_scores."""
    m, s = h.mu(Xg.ravel(), Sg.ravel())
    g = mu_true(Xg.ravel(), Sg.ravel())
    ok = (Sg.ravel() >= S_MIN) if feasible_only else np.ones(len(g), bool)
    e, v = (g - m)[ok], np.maximum(s[ok], SD_FLOOR) ** 2
    mae = float(np.mean(np.abs(e)))
    nll = float(np.mean(0.5 * np.log(2 * np.pi * v) + e ** 2 / (2 * v)))
    return mae, nll, m.reshape(Xg.shape), s.reshape(Xg.shape)


def closure_grid(box=BOX, n_grid=M_GRID, feasible_only=FEASIBLE_ONLY):
    """Uniform G = n_grid x n_grid grid over (X, S) with the withheld closure.

    The closure is the generating rate the hybrid does not know,
        g(X, S) = mu_max * S / (K_s + S) * 1 / (1 + k X^2)      [1/h]
    evaluated at every grid point. Returns (X, S, g) flattened.
    """
    (x_lo, x_hi), (s_lo, s_hi) = box
    if feasible_only:
        s_lo = max(s_lo, S_MIN)
    X, S = np.meshgrid(np.linspace(x_lo, x_hi, n_grid),
                       np.linspace(s_lo, s_hi, n_grid), indexing="ij")
    X, S = X.ravel(), S.ravel()
    return X, S, mu_true(X, S)


def closure_scores(model, grid):
    """Average grid MAE and average grid NLL of the withheld closure.

        MAE = (1/G) sum_p |g(x_p) - mu(x_p)|
        NLL = (1/G) sum_p [ 0.5 log(2 pi sigma^2(x_p))
                            + (g(x_p) - mu(x_p))^2 / (2 sigma^2(x_p)) ]

    mu(x_p) and sigma(x_p) are the predictive mean and standard deviation of the
    closure at grid point p. For the hybrid, mu = softplus(f) and sigma follows
    from the DKL's latent sigma by the delta method (models.Hybrid.mu).

    Only the hybrid HAS a closure. The black box learns the full right-hand side
    and never represents g, so it is not scored here -- see closure_arms().
    """
    X, S, g = grid
    m, s = model.mu(X, S)
    v = np.maximum(s, SD_FLOOR) ** 2
    e = g - m
    mae = float(np.mean(np.abs(e)))
    nll = float(np.mean(0.5 * np.log(2 * np.pi * v) + e ** 2 / (2 * v)))
    return mae, nll


# Closure metrics need a closure, which only the hybrid has: the "w/o hybrid"
# arm is therefore absent from fig_grid_metrics.
ARMS = [A_FULL, A_NOEX, A_RAND]


def metric_curves(cache=True):
    """Average grid MAE and NLL of the withheld closure at EVERY round, for every
    arm and repeat.

    Returns {label: (n_array, mae[rep, round], nll[rep, round])}. Cached to
    grid_metrics.json, since without checkpoints every model has to be refit.
    """
    path = f"{d}/grid_metrics.json"
    meta = {"feasible_only": FEASIBLE_ONLY, "m_grid": M_GRID, "box": [list(b) for b in BOX]}
    if cache and os.path.exists(path):
        raw = json.load(open(path))
        if raw.pop("_grid", None) == meta:
            return {k: (np.array(v["n"]), np.array(v["mae"]), np.array(v["nll"]))
                    for k, v in raw.items()}
        print("  grid_metrics.json was computed on a different grid -- recomputing")
    grid = closure_grid()
    out, src = {}, set()
    for pol, kind, label, _, _, _ in ARMS:
        ns, maes, nlls = None, [], []
        for fn in sorted(glob.glob(f"{d}/{pol}_s*.json")):
            seed = int(os.path.basename(fn).rsplit("_s", 1)[1].split(".")[0])
            rows = json.load(open(fn))
            data = [(np.array(r), np.array(y)) for r, y in rows[-1]["data"]]
            ns = [r["n"] for r in rows]
            a, b = [], []
            for n in ns:
                mdl, how = model_at(n, data, pol, kind, seed); src.add(how)
                mae, nll = closure_scores(mdl, grid)
                a.append(mae); b.append(nll)
            maes.append(a); nlls.append(b)
            print(f"    {label:24s} repeat {seed}: done", flush=True)
        out[label] = (np.array(ns), np.array(maes), np.array(nlls))
    if cache:
        dump = {k: {"n": v[0].tolist(), "mae": v[1].tolist(), "nll": v[2].tolist()}
                for k, v in out.items()}
        dump["_grid"] = meta
        json.dump(dump, open(path, "w"))
    if "refit" in src:
        print("  note: some models had no checkpoint and were refit cold")
    return out


def ci95(M):
    """Mean and half-width of the 95 % confidence interval of the mean, per
    column. Student-t with n_repeats - 1 degrees of freedom (n is small)."""
    from scipy import stats
    n = M.shape[0]
    m = M.mean(0)
    if n < 2:
        return m, np.zeros_like(m)
    return m, stats.t.ppf(0.975, n - 1) * M.std(0, ddof=1) / np.sqrt(n)


def fig_grid_metrics(curves=None):
    """Average grid MAE and average grid NLL of the withheld closure against the
    number of experiments, mean with a 95 % confidence band over repeats."""
    curves = curves or metric_curves()
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"wspace": 0.32})
    n_rep = 0
    panels = [(1, r"Average grid MAE of $g$ (h$^{-1}$)"),
              (2, "Average grid NLL of $g$ (nats)")]
    if FEASIBLE_ONLY:
        panels = [(k, lab.replace("grid", "feasible-grid")) for k, lab in panels]
    for i, (key, lab) in enumerate(panels):
        a = ax[i]
        vals = []
        for pol, kind, label, c, ls, mk in ARMS:
            if label not in curves:
                continue
            n, M = curves[label][0], curves[label][key]
            n_rep = max(n_rep, M.shape[0])
            m, h = ci95(M)
            vals.append(m)
            lo = np.maximum(m - h, 0.25 * m) if key == 1 else m - h   # (a) is log
            a.fill_between(n, lo, m + h, color=c, alpha=0.15 if ls == "-" else 0.07, lw=0)
            a.plot(n, m, ls, color=c, marker=mk, ms=3.2,
                   mfc=c if ls == "-" else "white", mew=0.8, label=label)
        for nc in cold_n():
            a.axvline(nc, color="0.8", lw=0.6, ls=":", zorder=0)
        # log scale only where the quantity is positive (NLL is routinely negative)
        if vals and min(v.min() for v in vals) > 0:
            a.set_yscale("log")
            a.set_ylim(bottom=max(PLOT_FLOOR, 0.2 * min(v.min() for v in vals)))
            a.yaxis.set_major_locator(mticker.LogLocator(subs=(1, 2, 5)))
            a.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
            a.yaxis.set_minor_formatter(mticker.NullFormatter())
        a.set_xlabel("Number of experiments"); a.set_ylabel(lab)
        n_max = int(curves[ARMS[0][2]][0].max()); step = 1 if n_max <= 9 else 2
        a.set_xticks(range(N_INIT, n_max + 1, step)); a.set_xlim(N_INIT - 0.4, n_max + 0.4)
        panel(a, "ab"[i])
    h_, l_ = ax[0].get_legend_handles_labels()
    if cold_n():
        h_.append(plt.Line2D([], [], color="0.7", lw=0.6, ls=":")); l_.append("cold DKL refit")
    bottom_legend(fig, h_, l_, ncol=4)
    save(fig, "fig_grid_metrics")
    (xl, xh), (sl, sh) = BOX
    sl = max(sl, S_MIN) if FEASIBLE_ONLY else sl
    print(f"\nWithheld closure g on a {M_GRID}x{M_GRID} grid, "
          f"X in [{xl:g}, {xh:g}], S in [{sl:g}, {sh:g}]"
          + ("  (feasible part only)" if FEASIBLE_ONLY else ""))
    print(f"{'arm':24s}{'MAE first -> last':>28s}{'NLL first -> last':>26s}")
    for pol, kind, label, _, _, _ in ARMS:
        if label not in curves:
            continue
        _, A, B = curves[label]
        ma, ha = ci95(A); mb, hb = ci95(B)
        print(f"{label:24s}{ma[0]:11.4f} -> {ma[-1]:.4f} ±{ha[-1]:.4f}"
              f"{mb[0]:11.2f} -> {mb[-1]:.2f} ±{hb[-1]:.2f}")
    print(f"  (mean, ± = 95 % CI of the mean over {n_rep} repeats; "
          f"the black box has no closure, so it is not scored here)")


def reach_box(pad=0.05):
    """(X, S) box spanned by every state observed in any arm of any repeat,
    padded by 5 %: the region the campaigns could actually reach."""
    pts = []
    for f in glob.glob(f"{d}/*_s*.json"):
        for _, y in json.load(open(f))[-1]["data"]:
            pts.append(np.array(y))
    pts = np.vstack(pts)
    x_hi = pts[:, 0].max(); s_hi = pts[:, 1].max()
    return (0.0, x_hi * (1 + pad)), (0.0, s_hi * (1 + pad))


def snapshot_ns(n_max):
    """Budgets shown in the kinetics figure and table: KINETICS_NS, keeping only
    those the campaign actually reached. If none survive (a shorter campaign),
    fall back to the initial design, midpoint and end."""
    ns = tuple(n for n in KINETICS_NS if N_INIT <= n <= n_max)
    return ns if ns else (N_INIT, N_INIT + (n_max - N_INIT) // 2, n_max)


def fig_kinetics(ns=None, box=BOX, n_grid=N_GRID):
    data = [(np.array(r), np.array(y)) for r, y in load("hybrid", SEED)[-1]["data"]]
    ns = ns or snapshot_ns(len(data))
    (x_lo, x_hi), (s_lo, s_hi) = box
    Xg, Sg = kinetics_grid(box, n_grid)
    mu_t = mu_true(Xg, Sg)

    fits, src = [], set()
    for n in ns:
        h, how = hybrid_at(n, data); src.add(how)
        mae, nll, m, s = grid_scores(h, Xg, Sg)
        fits.append((n, m, s, np.vstack([y for _, y in data[:n]]), mae, nll))

    mu_lv = np.arange(0, np.ceil(mu_t.max() * 20) / 20 + 1e-9, 0.05)
    # sd spans orders of magnitude between the first and last snapshot: log levels
    all_sd = np.concatenate([f[2].ravel() for f in fits])
    sd_lo = max(np.percentile(all_sd, 1), 1e-4); sd_hi = np.percentile(all_sd, 99.5)
    sd_lv = np.geomspace(sd_lo, sd_hi, 17)
    sd_norm = colors.LogNorm(sd_lo, sd_hi)

    nc = 1 + len(ns)
    fig, ax = plt.subplots(2, nc, figsize=(7.0, 3.9), sharex=True, sharey=True,
                           gridspec_kw={"hspace": 0.28, "wspace": 0.08})

    def deco(a, pts=None):
        a.axhspan(s_lo, S_MIN, facecolor="none", edgecolor=CON, hatch="////",
                  lw=0, alpha=0.5)
        a.axhline(S_MIN, color=CON, lw=0.8)
        if pts is not None:
            a.plot(pts[:, 0], pts[:, 1], "o", ms=1.8, mfc="white", mec="k", mew=0.3)
        a.set_xlim(x_lo, x_hi); a.set_ylim(s_lo, s_hi)

    # true kinetics
    cm_mu = ax[0, 0].contourf(Xg, Sg, mu_t, levels=mu_lv, cmap="viridis", extend="max")
    ax[0, 0].contour(Xg, Sg, mu_t, levels=mu_lv[::2], colors="k", linewidths=0.3, alpha=0.5)
    deco(ax[0, 0]); ax[0, 0].set_title(r"true $\mu$", loc="right")
    ax[0, 0].tick_params(labelbottom=True); ax[0, 0].set_xlabel(r"$X$ (g L$^{-1}$)")
    ax[1, 0].axis("off")
    ax[1, 0].text(0.38, 0.5, "ReHyDE:\nmean (top)\nstd. dev. (bottom)\n\nhatched: $S<S_{min}$", ha="center", va="center",
                  transform=ax[1, 0].transAxes, fontsize=6.5, color="0.4")

    for j, (n, m, s, pts, mae, nll) in enumerate(fits, start=1):
        ax[0, j].contourf(Xg, Sg, m, levels=mu_lv, cmap="viridis", extend="max")
        ax[0, j].contour(Xg, Sg, m, levels=mu_lv[::2], colors="k", linewidths=0.3, alpha=0.5)
        deco(ax[0, j], pts); ax[0, j].set_title(f"n = {n}", loc="right")
        cm_sd = ax[1, j].contourf(Xg, Sg, np.clip(s, sd_lo, None), levels=sd_lv,
                                  cmap="magma", norm=sd_norm, extend="max")
        deco(ax[1, j], pts)
        ax[0, j].text(0.97, 0.97, f"MAE {mae:.3f} h$^{{-1}}$",
                      transform=ax[0, j].transAxes, ha="right", va="top", fontsize=6,
                      bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.0))

    letters = iter("abcdefgh")
    for j in range(nc):
        panel(ax[0, j], next(letters))
    for j in range(1, nc):
        panel(ax[1, j], next(letters))
    for j in range(1, nc):
        ax[1, j].set_xlabel(r"$X$ (g L$^{-1}$)")
    ax[0, 0].set_ylabel(r"$S$ (g L$^{-1}$)"); ax[1, 1].set_ylabel(r"$S$ (g L$^{-1}$)")
    ax[1, 1].yaxis.set_tick_params(labelleft=True)
    cb1 = fig.colorbar(cm_mu, ax=ax[0, :], pad=0.01, fraction=0.02, aspect=12)
    cb1.set_label(r"$\mu$ (h$^{-1}$)"); cb1.outline.set_linewidth(0.5)
    cb2 = fig.colorbar(cm_sd, ax=ax[1, :], pad=0.01, fraction=0.02, aspect=12)
    cb2.set_label(r"sd of $\mu$ (h$^{-1}$)"); cb2.outline.set_linewidth(0.5)
    cb1.locator = mticker.MaxNLocator(5); cb1.update_ticks()
    cb2.locator = mticker.LogLocator(subs=(1, 2, 5)); cb2.update_ticks()
    cb2.formatter = mticker.FuncFormatter(lambda v, _: f"{v:g}"); cb2.update_ticks()
    save(fig, "fig_kinetics")
    if "refit" in src:
        print("  fig_kinetics: no checkpoints found -- hybrid refit cold on the "
              "first n experiments (not the campaign's warm-started models)")


# ------------------------------------------------------------------ trajectory
def feasible_test_set(n=N_TRAJ, seed=TRAJ_SEED):
    """n test recipes that are FEASIBLE on the plant (min S >= S_min over the
    horizon), with their noiseless plant trajectories. Drawn from a scrambled
    Sobol sequence over the recipe box and cached, so every arm, repeat and run
    is scored on exactly the same trajectories."""
    from scipy.stats.qmc import Sobol
    from plant import simulate, from_unit
    path = f"{d}/test_trajectories.npz"
    if os.path.exists(path):
        z = np.load(path)
        if len(z["R"]) == n:
            return z["R"], z["Y"]
    eng = Sobol(7, scramble=True, seed=seed)
    R, Y = [], []
    while len(R) < n:
        for u in eng.random(64):
            r = from_unit(u)
            y = simulate(r, noise=False)
            if y[:, 1].min() >= S_MIN:
                R.append(r); Y.append(y)
            if len(R) == n:
                break
    R, Y = np.array(R), np.array(Y)
    np.savez(path, R=R, Y=Y)
    return R, Y


@torch.no_grad()
def mc_trajectories(model, recipes, n_mc=N_MC, seed=0):
    """Propagate n_mc posterior realizations of the closure through the model.

    Each realization is a whitened inducing draw frozen for the whole rollout --
    the same object the training loss uses, not a mean +/- k*sigma surrogate --
    so the spread across realizations is the model's own predictive distribution
    over trajectories. Returns (n_mc, n_recipes, n_times, 2) at T_OBS.
    """
    from torchdiffeq import odeint
    from models import RK4_STEP, X_BOX, S_BOX
    from plant import T_OBS, N_CTRL, DT_CTRL
    R = torch.as_tensor(np.asarray(recipes))
    per_block = int(round(DT_CTRL / (T_OBS[1] - T_OBS[0])))
    gen = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n_mc):
        model.cache_sample(gen)
        z = R[:, :2].clone()
        obs = [z]
        for j in range(N_CTRL):
            D = R[:, 2 + j]
            rhs = lambda t, x: model.rhs(x[:, 0].clamp_min(1e-4), x[:, 1].clamp_min(0.0), D)
            tt = torch.as_tensor(T_OBS[j * per_block:(j + 1) * per_block + 1])
            sol = odeint(rhs, z, tt, method="rk4", options={"step_size": RK4_STEP})
            z = torch.stack([sol[-1][:, 0].clamp(1e-4, X_BOX[1]),
                             sol[-1][:, 1].clamp(0.0, S_BOX[1])], -1)
            obs += list(sol[1:])
        out.append(torch.stack(obs, 1))
    model.clear()
    return torch.stack(out).numpy()


def traj_scores(model, recipes, Ytrue, n_mc=N_MC, seed=0):
    """Trajectory MAE and coverage, per state.

    From the n_mc propagated realizations, the empirical mean and variance at
    every test trajectory i and time t are
        ybar^i_t = mean_s yhat^(i,s)_t,  Sigma^i_t = mean_s (yhat^(i,s)_t - ybar^i_t)^2.
    MAE is |y_true - ybar| averaged over trajectories and times, per state.
    Coverage is the fraction of (i, t) with the truth inside
    ybar +/- z * sqrt(Sigma), z the normal quantile for COV_LEVEL; it is
    returned per trajectory (averaged over t) so it can be shown as a violin.
    t = 0 is excluded: every realization starts at the known initial condition,
    so its variance is zero by construction.
    """
    from scipy import stats
    P = mc_trajectories(model, recipes, n_mc, seed)          # (S, N, T, 2)
    ybar = P.mean(0)
    var = np.maximum(P.var(0), VAR_FLOOR)
    z = stats.norm.ppf(0.5 + COV_LEVEL / 2)
    err = np.abs(Ytrue - ybar)[:, 1:, :]                     # drop t = 0
    inside = (err <= z * np.sqrt(var)[:, 1:, :])
    return (err[:, :, 0].mean(), err[:, :, 1].mean(),        # MAE of X, of S
            inside[:, :, 0].mean(1), inside[:, :, 1].mean(1))  # per-trajectory cov


TRAJ_ARMS = ALL_ARMS


def traj_curves(cache=True):
    """MAE (per state) at every round and per-trajectory coverage at the final
    round, for all four arms and every repeat. Cached to traj_metrics.npz."""
    path = f"{d}/traj_metrics.npz"
    if cache and os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return z["curves"].item()
    R, Y = feasible_test_set()
    print(f"  scoring {len(R)} feasible test trajectories with {N_MC} "
          f"posterior realizations per model", flush=True)
    out, src = {}, set()
    for pol, kind, label, _, _, _ in TRAJ_ARMS:
        ns, mx, ms, cx, cs = None, [], [], [], []
        for fn in sorted(glob.glob(f"{d}/{pol}_s*.json")):
            seed = int(os.path.basename(fn).rsplit("_s", 1)[1].split(".")[0])
            rows = json.load(open(fn))
            data = [(np.array(r), np.array(y)) for r, y in rows[-1]["data"]]
            ns = [r["n"] for r in rows]
            a, b = [], []
            for n in ns:
                mdl, how = model_at(n, data, pol, kind, seed); src.add(how)
                ex, es, gx, gs = traj_scores(mdl, R, Y, seed=seed)
                a.append(ex); b.append(es)
                if n == ns[-1]:
                    cx.append(gx); cs.append(gs)
            mx.append(a); ms.append(b)
            print(f"    {label:24s} repeat {seed}: done", flush=True)
        out[label] = {"n": np.array(ns), "mae_X": np.array(mx), "mae_S": np.array(ms),
                      "cov_X": np.concatenate(cx), "cov_S": np.concatenate(cs)}
    if cache:
        np.savez(path, curves=np.array(out, dtype=object))
    if "refit" in src:
        print("  note: some models had no checkpoint and were refit cold")
    return out


def fig_trajectory(curves=None):
    """Four panels on the same 100 feasible test trajectories:
    (a, b) trajectory MAE of X and S against the number of experiments, mean
    with a 95 % confidence band over repeats; (c, d) the distribution across
    trajectories of the coverage of the propagated predictive interval, at the
    final budget."""
    curves = curves or traj_curves()
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 4.6),
                           gridspec_kw={"hspace": 0.45, "wspace": 0.28})

    for i, (key, lab) in enumerate([("mae_X", r"Trajectory MAE of $X$ (g L$^{-1}$)"),
                                    ("mae_S", r"Trajectory MAE of $S$ (g L$^{-1}$)")]):
        a = ax[0, i]
        for pol, kind, label, c, ls, mk in TRAJ_ARMS:
            if label not in curves:
                continue
            n, M = curves[label]["n"], curves[label][key]
            m, h = ci95(M)
            a.fill_between(n, np.maximum(m - h, 0.25 * m), m + h, color=c,
                           alpha=0.15 if ls == "-" else 0.07, lw=0)
            a.plot(n, m, ls, color=c, marker=mk, ms=3.0,
                   mfc=c if ls == "-" else "white", mew=0.8, label=label)
        for nc in cold_n():
            a.axvline(nc, color="0.8", lw=0.6, ls=":", zorder=0)
        a.set_yscale("log")
        a.yaxis.set_major_locator(mticker.LogLocator(subs=(1, 2, 5)))
        a.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        a.yaxis.set_minor_formatter(mticker.NullFormatter())
        a.set_xlabel("Number of experiments"); a.set_ylabel(lab)
        n_max = int(n.max()); step = 1 if n_max <= 9 else 2
        a.set_xticks(range(N_INIT, n_max + 1, step)); a.set_xlim(N_INIT - 0.4, n_max + 0.4)
        panel(a, "ab"[i])

    labels = [l for _, _, l, _, _, _ in TRAJ_ARMS if l in curves]
    cols = [c for _, _, l, c, _, _ in TRAJ_ARMS if l in curves]
    styles = [ls for _, _, l, _, ls, _ in TRAJ_ARMS if l in curves]
    for i, (key, lab) in enumerate([("cov_X", r"Coverage for $X$"),
                                    ("cov_S", r"Coverage for $S$")]):
        a = ax[1, i]
        data = [100 * curves[l][key] for l in labels]
        parts = a.violinplot(data, positions=range(len(data)), widths=0.75,
                             showextrema=False, showmedians=False)
        for body, c, ls in zip(parts["bodies"], cols, styles):
            body.set_facecolor(c); body.set_edgecolor(c); body.set_linewidth(0.8)
            body.set_alpha(0.45 if ls == "-" else 0.2)
            # coverage is a fraction: clip the KDE's tails to [0, 100]
            v = body.get_paths()[0].vertices
            v[:, 1] = np.clip(v[:, 1], 0.0, 100.0)
        for j, (v, c) in enumerate(zip(data, cols)):
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            a.vlines(j, q1, q3, color=c, lw=3.0, zorder=3)
            a.plot(j, med, "o", ms=3.2, mfc="white", mec=c, mew=1.0, zorder=4)
        a.axhline(100 * COV_LEVEL, color=CON, lw=0.9, ls="--", zorder=1)
        a.set_xticks(range(len(labels)))
        a.set_xticklabels([l.replace("ReHyDE w/o ", "w/o\n") for l in labels],
                          fontsize=6.5)
        a.set_ylim(-3, 108); a.set_ylabel(lab + " (%)")
        a.set_yticks([0, 20, 40, 60, 80, 100])
        panel(a, "cd"[i])

    h_, l_ = ax[0, 0].get_legend_handles_labels()
    if cold_n():
        h_.append(plt.Line2D([], [], color="0.7", lw=0.6, ls=":")); l_.append("cold DKL refit")
    h_.append(plt.Line2D([], [], color=CON, lw=0.9, ls="--"))
    l_.append(f"nominal {100 * COV_LEVEL:g} % coverage")
    bottom_legend(fig, h_, l_, ncol=3)
    save(fig, "fig_trajectory")

    n_rep = curves[labels[0]]["mae_X"].shape[0]
    print(f"\nTrajectory metrics on {N_TRAJ} feasible test trajectories, "
          f"{N_MC} posterior realizations, {100 * COV_LEVEL:g} % interval")
    print(f"{'arm':24s}{'MAE X first -> last':>26s}{'MAE S first -> last':>26s}"
          f"{'cov X':>9s}{'cov S':>8s}")
    for _, _, label, _, _, _ in TRAJ_ARMS:
        if label not in curves:
            continue
        cv = curves[label]
        ma, ha = ci95(cv["mae_X"]); mb, hb = ci95(cv["mae_S"])
        print(f"{label:24s}{ma[0]:9.4f} -> {ma[-1]:.4f} ±{ha[-1]:.4f}"
              f"{mb[0]:9.4f} -> {mb[-1]:.4f} ±{hb[-1]:.4f}"
              f"{100 * cv['cov_X'].mean():8.1f}%{100 * cv['cov_S'].mean():7.1f}%")
    print(f"  (MAE: mean, ± = 95 % CI over {n_rep} repeats; coverage: mean over "
          f"trajectories at the final budget)")


# ------------------------------------------------------------------ grid metrics
def kinetics_metrics(arms=tuple((a[0], a[2]) for a in (A_FULL, A_NOEX, A_RAND))):
    """Grid RMSE and NLL of the hybrid's learned mu for EVERY repeat, at the
    same snapshots as the figure; printed and written to kinetics_metrics.csv."""
    Xg, Sg = kinetics_grid()
    rows, src = [], set()
    for pol, label in arms:
        for f in sorted(glob.glob(f"{d}/{pol}_s*.json")):
            seed = int(os.path.basename(f).rsplit("_s", 1)[1].split(".")[0])
            data = [(np.array(r), np.array(y)) for r, y in json.load(open(f))[-1]["data"]]
            for n in snapshot_ns(len(data)):
                h, how = hybrid_at(n, data, pol, seed); src.add(how)
                mae, nll, _, _ = grid_scores(h, Xg, Sg)
                rows.append((label, seed, n, mae, nll))
    with open(f"{d}/kinetics_metrics.csv", "w") as fh:
        fh.write("arm,repeat,n_experiments,mae_per_h,nll\n")
        for r in rows:
            fh.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.5f},{r[4]:.4f}\n")
    (xl, xh), (sl, sh) = BOX
    print(f"\nHybrid kinetics on the grid X in [{xl:g}, {xh:g}], S in [{max(sl, S_MIN):g}, {sh:g}] "
          f"({N_GRID}x{N_GRID}, S >= S_min)")
    print(f"{'arm':24s}{'n':>4s}{'MAE (1/h)':>20s}{'NLL':>20s}   per repeat MAE / NLL")
    for _, label in arms:
        for n in sorted({r[2] for r in rows if r[0] == label}):
            sel = [r for r in rows if r[0] == label and r[2] == n]
            R = np.array([r[3] for r in sel]); L = np.array([r[4] for r in sel])
            per = "  ".join(f"s{r[1]}:{r[3]:.3f}/{r[4]:.2f}" for r in sel)
            print(f"{label:24s}{n:4d}{R.mean():12.4f} ± {R.std():.4f}"
                  f"{L.mean():12.3f} ± {L.std():.3f}   {per}")
    print(f"  (mean ± sd over {len({r[1] for r in rows})} repeats; "
          f"written to {d}/kinetics_metrics.csv)")
    if "refit" in src:
        print("  note: no checkpoints found for some models -- those were refit cold on "
              "the first n experiments")


if __name__ == "__main__":
    clean_outputs()
    fig_phase(); fig_kinetics(); fig_grid_metrics(); fig_trajectory()
    kinetics_metrics()
    print(f"wrote fig_phase, fig_kinetics, fig_grid_metrics, fig_trajectory (.pdf, .png) in {d} for repeat {SEED}")