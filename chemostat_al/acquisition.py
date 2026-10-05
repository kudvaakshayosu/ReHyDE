"""Constrained exploration: maximise int ||sigma_g|| dt s.t. P(S(t) >= S_min) >= 1-alpha.

Search, in the unit-scaled 7-D recipe box:
  1. global screen   N_SCREEN scrambled Sobol recipes, scored in one batch
  2. local refine    Nelder-Mead with an exact L1 penalty, multistarted from
                     the N_START best MUTUALLY DISTINCT screened candidates;
                     the starts advance in lockstep so every iteration is one
                     batched rollout
  3. verify          every candidate is re-checked with the discrete chance
                     constraint; the best feasible one is returned

The chance constraint is estimated from K-1 posterior-sample trajectories. As
a fraction of samples it is piecewise constant in the recipe -- a simplex gets
no signal from it -- so the refinement uses a continuous margin that agrees
with the fraction test exactly at the feasibility boundary (see `margin`).
The posterior-sample directions are drawn once per call and reused for every
evaluation (common random numbers), so the objective is deterministic.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol
from models import rollout
from plant import LB, UB, S_MIN, from_unit, to_unit

ALPHA = 0.1

# ---- search settings ---------------------------------------------------------
N_SCREEN = 1024      # Sobol candidates in the global screen
N_START = 4          # Nelder-Mead starts (distinct basins)
MIN_SEP = 0.15       # minimum unit-box distance between starts
MAX_ITER = 150       # Nelder-Mead iterations (all starts advance together)
SIMPLEX = 0.05       # initial simplex edge, as a fraction of each bound range
RHO = 50.0           # L1 penalty per g/L of constraint violation (normalised units)
RHO_BOX = 1e3        # quadratic penalty for leaving the box


def make_eps(model, K, rng):
    """Posterior-sample directions for the chance constraint (row 0 = mean)."""
    e = rng.standard_normal((K - 1, 2))
    return np.vstack([np.zeros((1, 2)), e])


def score(model, recipes, eps):
    traj, isig = rollout(model, recipes, eps)
    S = traj[:, 1:, :, 1]                         # samples only (exclude mean)
    p_ok = (S >= S_MIN).mean(1)                    # (N, 11) per-time probability
    p_min = p_ok.min(1)
    acq = isig[:, 0]                               # int sigma along mean trajectory
    return acq, p_min, traj


def margin(traj, alpha=ALPHA):
    """Continuous chance-constraint margin [g/L]; feasible iff margin >= 0.

    For each sampling time t, the alpha-quantile over the posterior samples of
    S_s(t) - S_min; the margin is its minimum over t. With the default 11
    samples and alpha = 0.1, np.quantile's linear rule puts the 0.1-quantile
    exactly on the second-smallest sample, so margin >= 0  <=>  at most one of
    11 samples violates at every t  <=>  the fraction test p >= 1 - alpha.
    Same feasible set, but continuous in the recipe.
    """
    S = traj[:, 1:, :, 1]                                  # (N, K-1, T)
    return np.quantile(S - S_MIN, alpha, axis=1).min(1)    # (N,)


def _distinct_starts(U, key, n, sep=MIN_SEP):
    """Greedy: best-ranked candidates, each at least `sep` (unit-box Euclidean
    distance) from those already chosen, so the starts sit in different basins
    instead of all polishing the same point."""
    chosen = []
    for i in np.argsort(-key):
        if all(np.linalg.norm(U[i] - U[j]) >= sep for j in chosen):
            chosen.append(i)
            if len(chosen) == n:
                break
    return chosen


def _objective(model, eps, U, a_ref):
    """Penalised objective for a BATCH of unit-box points, in one rollout:
        -acq/a_ref + RHO * max(0, -margin) + RHO_BOX * ||u - clip(u)||^2
    The L1 penalty is exact: above a finite RHO the constrained and penalised
    optima coincide. Returns (f, acq, margin) for each row of U."""
    Uc = np.clip(U, 0.0, 1.0)
    traj, isig = rollout(model, from_unit(Uc), eps)
    acq, g = isig[:, 0], margin(traj)
    f = -acq / a_ref + RHO * np.maximum(0.0, -g) + RHO_BOX * ((U - Uc) ** 2).sum(1)
    return f, acq, g


def _nelder_mead_batched(model, eps, U0, a_ref, max_iter=MAX_ITER,
                         xatol=1e-3, fatol=1e-5):
    """Multistart Nelder-Mead run in lockstep.

    Standard coefficients (reflection 1, expansion 2, contraction 1/2, shrink
    1/2). Every iteration evaluates, for every start at once, all four trial
    points -- reflection, expansion, outside and inside contraction -- in ONE
    batched rollout, then applies the usual NM decision to each simplex. This
    'speculative' evaluation costs a few extra points per iteration but turns
    thousands of sequential ~20 ms calls into ~one call per iteration, which is
    what makes the refinement affordable. A simplex that meets the tolerances
    is frozen. Returns the best vertex of every simplex and the evaluation count.
    """
    n, d = U0.shape
    # explicit initial simplex (scipy's default perturbs by 5 % of the VALUE,
    # i.e. ~0 at the lower bound, where D often sits)
    S = np.repeat(U0[:, None, :], d + 1, axis=1)                 # (n, d+1, d)
    for i in range(d):
        step = np.where(U0[:, i] + SIMPLEX <= 1.0, SIMPLEX, -SIMPLEX)
        S[:, i + 1, i] += step
    F = _objective(model, eps, S.reshape(-1, d), a_ref)[0].reshape(n, d + 1)
    n_eval = n * (d + 1)
    active = np.ones(n, bool)
    for _ in range(max_iter):
        o = np.argsort(F, 1)
        S = np.take_along_axis(S, o[:, :, None], 1); F = np.take_along_axis(F, o, 1)
        conv = (np.abs(S - S[:, :1]).max((1, 2)) < xatol) & (F[:, -1] - F[:, 0] < fatol)
        active &= ~conv
        idx = np.flatnonzero(active)
        if len(idx) == 0:
            break
        c = S[idx, :-1].mean(1); w = S[idx, -1]
        trial = np.stack([c + (c - w), c + 2 * (c - w),            # reflect, expand
                          c + 0.5 * (c - w), c - 0.5 * (c - w)], 1)  # contract out, in
        Ft = _objective(model, eps, trial.reshape(-1, d), a_ref)[0].reshape(-1, 4)
        n_eval += trial.shape[0] * 4
        shrink = []
        for k, s in enumerate(idx):
            fr, fe, fo, fi = Ft[k]
            if fr < F[s, 0]:                                  # better than best
                S[s, -1], F[s, -1] = (trial[k, 1], fe) if fe < fr else (trial[k, 0], fr)
            elif fr < F[s, -2]:                               # better than 2nd worst
                S[s, -1], F[s, -1] = trial[k, 0], fr
            elif fr < F[s, -1]:                               # outside contraction
                if fo <= fr:
                    S[s, -1], F[s, -1] = trial[k, 2], fo
                else:
                    shrink.append(s)
            elif fi < F[s, -1]:                               # inside contraction
                S[s, -1], F[s, -1] = trial[k, 3], fi
            else:
                shrink.append(s)
        if shrink:                                            # shrink towards best
            sh = np.array(shrink)
            S[sh, 1:] = S[sh, :1] + 0.5 * (S[sh, 1:] - S[sh, :1])
            F[sh, 1:] = _objective(model, eps, S[sh, 1:].reshape(-1, d),
                                   a_ref)[0].reshape(len(sh), d)
            n_eval += len(sh) * d
    best = np.argmin(F, 1)
    return np.clip(S[np.arange(n), best], 0.0, 1.0), n_eval


def select(model, rng, n_screen=N_SCREEN, n_start=N_START, K=12, info=None):
    """ReHyDE acquisition: argmax int ||sigma_g|| dt s.t. the chance constraint.

    Returns (recipe, acquisition value, constraint probability). If `info` is a
    dict it is filled with search diagnostics (screen best, refined best, evals).
    """
    eps = make_eps(model, K, rng)
    U = Sobol(len(LB), scramble=True, seed=int(rng.integers(1 << 31))).random(n_screen)
    acq, p, traj = score(model, from_unit(U), eps)
    g = margin(traj)
    feas = g >= 0.0

    # rank: feasible by acquisition, then infeasible by margin (feasibility search)
    key = np.where(feas, 1e6 + acq, g)
    starts = _distinct_starts(U, key, n_start)
    a_ref = max(float(acq[feas].max()) if feas.any() else float(acq.max()), 1e-12)

    refined, n_nm = _nelder_mead_batched(model, eps, U[starts], a_ref)
    cand = np.vstack([U[starts], refined])      # keep the starts: never worse than the screen
    n_eval = n_screen + n_nm

    # verify every endpoint with the same samples, then pick
    a2, p2, t2 = score(model, from_unit(cand), eps)
    g2 = margin(t2)
    ok = g2 >= 0.0
    j = int(np.argmax(np.where(ok, a2, -np.inf))) if ok.any() else int(np.argmax(g2))
    if info is not None:
        info.update(screen_best=float(acq[feas].max()) if feas.any() else np.nan,
                    refined_best=float(a2[j]), n_eval=int(n_eval),
                    n_feasible_screen=int(feas.sum()), feasible=bool(ok[j]))
    return from_unit(cand[j]), float(a2[j]), float(p2[j])


def select_constrained_random(model, rng, n_cand=256, K=12):
    """Ablation of the exploration objective: a UNIFORM design restricted to the
    predicted feasible region.

    Draws n_cand recipes uniformly over the box, keeps those satisfying the same
    chance constraint P(S(t_k) >= S_min) >= 1 - ALPHA that `select` enforces, and
    returns one of them uniformly at random -- i.e. rejection sampling from the
    uniform design distribution on the model's feasible set. If none is feasible
    it falls back to the highest constraint probability, exactly as `select`
    does, so the arm never stalls.

    Isolating the information objective is the point: this arm keeps the region
    constraint (Re) and the hybrid closure (Hy) but replaces the
    int ||sigma_g|| dt objective (E) with a uniform draw.
    """
    eps = make_eps(model, K, rng)
    R = from_unit(rng.random((n_cand, len(LB))))
    acq, p, _ = score(model, R, eps)
    feas = np.flatnonzero(p >= 1 - ALPHA)
    i = int(rng.choice(feas)) if len(feas) else int(np.argmax(p))
    return R[i], acq[i], p[i]
