"""Hybrid (mass balances + DKL for mu) and black-box (DKLs for dX/dt, dS/dt).

Both are trained on trajectories through the ODE (fit.py), as in CHO -- no
derivative targets. Rollouts for acquisition, constraint and evaluation use
torchdiffeq.odeint (RK4), one call per 2 h D-block, with the acquisition
integral carried as a third state dI/dt = ||sigma_g||.
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchdiffeq import odeint
from dkl import DKL
from fit import train
from plant import T_OBS, Y, S_IN, N_CTRL, DT_CTRL

X_BOX, S_BOX, D_BOX = (0.0, 5.0), (0.0, 12.0), (0.02, 0.3)

# What the hybrid's acquisition integrates along the trajectory, int ||sigma|| dt:
#   "closure"  sigma of the latent closure f             [1/h]    (original)
#   "induced"  sigma the closure induces on the state derivatives, by the delta
#              method through mu = softplus(f) and the known balances:
#              sigma_xdot = sigma_f * sigmoid(f) * X * sqrt(1 + 1/Y^2)   [g/L/h]
#              -- the same units the black box integrates, and zero where an
#              error in mu cannot move the dynamics (X -> 0).
ACQ_SIGMA = __import__("os").environ.get("ACQ_SIGMA", "closure")   # "closure" | "induced"
RK4_STEP = 0.5
BB_SCALE = (1.0, 3.0)        # output scale of the black-box latents [g/L/h]


def _obs_inputs(data, with_D):
    Xs = []
    for r, y in data:
        if with_D:
            from plant import D_at
            Xs.append(np.c_[y, D_at(r, T_OBS)])
        else:
            Xs.append(y)
    return torch.as_tensor(np.vstack(Xs))


class _Base(nn.Module):
    def cache_mean(self):
        for m in self.fs: m.cache_mean()

    def cache_sample(self, gen):
        for m in self.fs: m.cache_sample(gen)

    def clear(self):
        for m in self.fs: m.clear()

    def kl(self):
        return sum(m.kl() for m in self.fs)

    def fit(self, data, seed=0, warm=None):
        if warm is None:
            self.build(data, seed)
        else:                     # warm start: continue from the previous model
            self.load_state_dict(warm.state_dict(), strict=False)
            self.copy_from(warm)
        train(self, data, cold=warm is None, seed=seed)
        self.eval()
        return self


class Hybrid(_Base):
    name = "hybrid"

    def build(self, data, seed):
        self.f = DKL([X_BOX[0], S_BOX[0]], [X_BOX[1], S_BOX[1]],
                     _obs_inputs(data, False), len(data), seed)
        self.f.gp.mean_module.initialize(constant=0.0)   # softplus(0) = 0.69
        self.fs = [self.f]

    def copy_from(self, warm):
        self.f = warm.f; self.fs = [self.f]

    # training rhs: current cached realisation
    def rhs(self, X, S, D):
        mu = F.softplus(self.f.realisation(torch.stack([X, S], -1)))
        return torch.stack([(mu - D) * X, -mu * X / Y + D * (S_IN - S)], -1)

    # rollout rhs: predictive mean + eps * sd
    def g(self, X, S, D, eps):
        m, s = self.f.predict(torch.stack([X, S], -1))
        f = m + eps[:, 0] * s
        mu = F.softplus(f)
        if ACQ_SIGMA == "induced":
            sig = s * torch.sigmoid(f) * X * (1.0 + 1.0 / Y ** 2) ** 0.5
        else:
            sig = s
        return (mu - D) * X, -mu * X / Y + D * (S_IN - S), sig

    def mu(self, X, S):
        """Learned kinetics: mean and (delta-method) sd of mu = softplus(f)."""
        m, s = self.f.predict(torch.stack([torch.as_tensor(X), torch.as_tensor(S)], -1))
        return F.softplus(m).numpy(), (s * torch.sigmoid(m)).numpy()


class BlackBox(_Base):
    name = "blackbox"

    def build(self, data, seed):
        lb, ub = [X_BOX[0], S_BOX[0], D_BOX[0]], [X_BOX[1], S_BOX[1], D_BOX[1]]
        Z = _obs_inputs(data, True)
        self.fx = DKL(lb, ub, Z, len(data), seed)
        self.fs_ = DKL(lb, ub, Z, len(data), seed + 1)
        self.fs = [self.fx, self.fs_]

    def copy_from(self, warm):
        self.fx, self.fs_ = warm.fx, warm.fs_; self.fs = [self.fx, self.fs_]

    def rhs(self, X, S, D):
        Z = torch.stack([X, S, D], -1)
        return torch.stack([BB_SCALE[0] * self.fx.realisation(Z),
                            BB_SCALE[1] * self.fs_.realisation(Z)], -1)

    def g(self, X, S, D, eps):
        Z = torch.stack([X, S, D], -1)
        mx, sx = self.fx.predict(Z); ms, ss = self.fs_.predict(Z)
        sx, ss = BB_SCALE[0] * sx, BB_SCALE[1] * ss
        return (BB_SCALE[0] * mx + eps[:, 0] * sx, BB_SCALE[1] * ms + eps[:, 1] * ss,
                (sx ** 2 + ss ** 2).sqrt())


# --------------------------------------------------------------------------- #
# checkpoints
# --------------------------------------------------------------------------- #
def save_model(model, path, **meta):
    """Save a trained model with everything needed to rebuild it."""
    torch.save({"kind": model.name, "state_dict": model.state_dict(), **meta}, path)


def _empty_dkl(sd, prefix, lb, ub):
    M = sd[f"{prefix}.gp.variational_strategy.inducing_points"].shape[0]
    return DKL(lb, ub, torch.zeros(M, len(lb)), n_recipes=M)


def load_model(path):
    """Rebuild a saved model from its checkpoint alone.

    The inducing count M is read from the state_dict, not from a dataset:
    warm starts keep the inducing set of the last COLD fit, so M need not
    equal 5 x (recipes at save time).
    Returns (model, metadata).
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    if ck["kind"] == "hybrid":
        mdl = Hybrid()
        mdl.f = _empty_dkl(sd, "f", [X_BOX[0], S_BOX[0]], [X_BOX[1], S_BOX[1]])
        mdl.fs = [mdl.f]
    else:
        lb, ub = [X_BOX[0], S_BOX[0], D_BOX[0]], [X_BOX[1], S_BOX[1], D_BOX[1]]
        mdl = BlackBox()
        mdl.fx = _empty_dkl(sd, "fx", lb, ub); mdl.fs_ = _empty_dkl(sd, "fs_", lb, ub)
        mdl.fs = [mdl.fx, mdl.fs_]
    mdl.load_state_dict(sd)
    mdl.eval()
    return mdl, {k: v for k, v in ck.items() if k != "state_dict"}


@torch.no_grad()
def rollout(model, recipes, eps):
    """recipes (N,7); eps (K,2) with eps[0] = 0 (mean model).

    Each sampled trajectory applies a fixed eps for the whole run (comonotone
    approximation of a posterior sample), used for the chance constraint.
    State [X, S, I] with dI/dt = ||sigma_g||, so the acquisition is integrated
    by the same solver. Returns traj (N,K,11,2) at T_OBS and I(t_f) (N,K).
    """
    R = torch.as_tensor(np.asarray(recipes)); E = torch.as_tensor(np.asarray(eps))
    N, K = len(R), len(E)
    Rr = R.repeat_interleave(K, 0); Er = E.repeat(N, 1)
    z = torch.stack([Rr[:, 0], Rr[:, 1], torch.zeros(N * K)], -1)
    obs = [z[:, :2]]
    per_block = int(round(DT_CTRL / (T_OBS[1] - T_OBS[0])))
    for j in range(N_CTRL):
        D = Rr[:, 2 + j]

        def f(t, z):
            X = z[:, 0].clamp_min(1e-4); S = z[:, 1].clamp_min(0.0)
            fx, fs, sig = model.g(X, S, D, Er)
            return torch.stack([fx, fs, sig], -1)

        tt = torch.as_tensor(T_OBS[j * per_block:(j + 1) * per_block + 1])
        sol = odeint(f, z, tt, method="rk4", options={"step_size": RK4_STEP})
        z = sol[-1]
        z = torch.stack([z[:, 0].clamp(1e-4, X_BOX[1]), z[:, 1].clamp(0, S_BOX[1]), z[:, 2]], -1)
        obs += [s[:, :2] for s in sol[1:]]
    traj = torch.stack(obs, 1).reshape(N, K, len(T_OBS), 2).numpy()
    return traj, z[:, 2].reshape(N, K).numpy()
