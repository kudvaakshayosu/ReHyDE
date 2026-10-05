"""Trajectory training of the DKL models through torchdiffeq, as in CHO.

Multiple shooting: every observation interval (1 h, D constant) is a segment
that starts at the MEASURED state and is integrated (RK4, 0.5 h steps) to the
next measurement, where the residual is taken. All segments of all recipes are
integrated in one batched odeint call.

  phase 1  deterministic warm-up on the posterior-mean realisation
  phase 2  sampled realisations (n_mc frozen inducing draws per step) + KL,
           KL weight annealed 0 -> 1 over the first 30 %, GP lr x 0.3; network
           frozen, variational covariance started at S_INIT * I (cold fits)
Adam: network lr 3e-3, GP lr 1e-2, gradient clip 5.
Cold fit: EPOCHS; warm fit continues from the previous model with WARM_EPOCHS.
"""
import numpy as np
import torch
from torchdiffeq import odeint
from plant import T_OBS, D_at

LR, LR_GP, CLIP, KL_FRAC, S_INIT = 3e-3, 1e-2, 5.0, 0.3, 0.05
EPOCHS, WARM_EPOCHS, N_MC = (600, 200), (150, 60), 2
RK4_STEP = 0.5
# Residuals are whitened by the measurement covariance Sigma = diag(SIG_X^2,
# SIG_S^2) of plant.py, so the fit term is the Gaussian log-likelihood of the
# observation model and is weighted against KL / n_data exactly as in the ELBO.
from plant import SIGMA


def _clear_gp_cache(model):
    """gpytorch memoises intermediates on the variational strategy; drop them
    between steps or the next backward hits a freed graph (as in CHO)."""
    for m in model.modules():
        for a in list(vars(m)):
            if a.startswith("_memoize_cache"):
                setattr(m, a, {})


def segments(data):
    """Start states, end states and D for every 1 h observation interval."""
    x0, x1, D = [], [], []
    for r, y in data:
        x0.append(y[:-1]); x1.append(y[1:]); D.append(D_at(r, T_OBS[:-1]))
    return (torch.as_tensor(np.vstack(x0)), torch.as_tensor(np.vstack(x1)),
            torch.as_tensor(np.concatenate(D)))


def seg_loss(model, seg):
    x0, x1, D = seg
    rhs = lambda t, x: model.rhs(x[:, 0].clamp_min(1e-4), x[:, 1].clamp_min(0.0), D)
    pred = odeint(rhs, x0, torch.tensor([0.0, T_OBS[1] - T_OBS[0]]),
                  method="rk4", options={"step_size": RK4_STEP})[-1]
    r = (pred - x1) / torch.as_tensor(SIGMA)
    return 0.5 * (r ** 2).mean()


def train(model, data, cold, seed=0):
    seg = segments(data)
    gen = torch.Generator().manual_seed(seed)
    ep = EPOCHS if cold else WARM_EPOCHS
    nets = [p for n, p in model.named_parameters() if ".feature." in n or n.startswith("feature")]
    gps = [p for n, p in model.named_parameters() if not (".feature." in n or n.startswith("feature"))]
    n_data = seg[1].numel()

    opt = torch.optim.Adam([{"params": nets, "lr": LR}, {"params": gps, "lr": LR_GP}])
    for _ in range(ep[0]):
        opt.zero_grad(); _clear_gp_cache(model)
        model.cache_mean()
        loss = seg_loss(model, seg)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP); opt.step()

    # Phase 2 calibrates the GP's uncertainty on the features learned in phase 1.
    # Deviations from CHO, both needed here (see summary.tex): at a cold fit the
    # whitened variational covariance starts at S_INIT * I instead of the prior
    # (prior draws otherwise diverge and wreck the mean fit), and the network is
    # frozen in phase 2.
    if cold:
        with torch.no_grad():
            for d in model.fs:
                q = d.gp.variational_strategy._variational_distribution
                q.chol_variational_covar.copy_(S_INIT * torch.eye(len(q.variational_mean)))
    for p in nets:
        p.requires_grad_(False)
    opt = torch.optim.Adam([{"params": gps, "lr": 0.3 * LR_GP}])
    n_anneal = max(1, int(KL_FRAC * ep[1]))
    for e in range(ep[1]):
        opt.zero_grad(); _clear_gp_cache(model)
        fit, used = 0.0, 0
        for _ in range(N_MC):
            model.cache_sample(gen)
            l = seg_loss(model, seg)
            if torch.isfinite(l):            # drop divergent draws, as in CHO
                fit = fit + l; used += 1
        if used == 0:
            continue
        beta = min(1.0, (e + 1) / n_anneal) if cold else 1.0
        loss = fit / used + beta * model.kl() / n_data
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP); opt.step()
    for p in nets:
        p.requires_grad_(True)
    model.clear(); _clear_gp_cache(model)
    return model
