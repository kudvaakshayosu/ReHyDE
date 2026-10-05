"""True chemostat plant (from Dynamics_simulator.py 'chemo_stat') and recipe helpers.

Recipe (7-D): x = [X0, S0, D1..D5]; each D_j is held for 2 h, horizon 10 h.

Observation model (matches the paper's formulation):
    y_tk = C x_tk + eps_tk,   eps_tk ~ N(0, Sigma),   Sigma = diag(SIG_X^2, SIG_S^2)
Both states are observed (C = I), with a CONSTANT diagonal covariance -- additive,
homoscedastic, independent across states and sampling times.
"""
import numpy as np
from scipy.integrate import solve_ivp

Y, S_IN = 0.5, 10.0
MU_MAX, KS, K_INH = 0.8, 0.1, 0.3

T_END, DT_OBS, DT_CTRL = 10.0, 1.0, 2.0
T_OBS = np.arange(0.0, T_END + 1e-9, DT_OBS)          # 11 points
N_CTRL = int(T_END / DT_CTRL)                          # 5 D values
S_MIN = 0.5

# measurement noise: Sigma = diag(SIG_X^2, SIG_S^2) [g/L]
SIG_X, SIG_S = 0.01, 0.02
SIGMA = np.array([SIG_X, SIG_S])

LB = np.array([0.05, 2.0] + [0.02] * N_CTRL)
UB = np.array([1.50, 9.0] + [0.30] * N_CTRL)
NAMES = ["X0", "S0"] + [f"D{j+1}" for j in range(N_CTRL)]


def mu_true(X, S):
    return MU_MAX * S / (KS + S) / (1.0 + K_INH * X ** 2)


def D_at(recipe, t):
    """Dilution rate active at time t (piecewise constant, 2 h blocks)."""
    j = np.minimum((np.asarray(t) / DT_CTRL).astype(int), N_CTRL - 1)
    return np.asarray(recipe)[2 + j]


def simulate(recipe, noise=True, rng=None):
    """Observed states at T_OBS, shape (11, 2) = [X, S].

    noise=True adds eps ~ N(0, Sigma); noise=False returns the clean plant
    trajectory, which is what the test sets are scored against.
    """
    x = [recipe[0], recipe[1]]
    out = [list(x)]
    for k in range(len(T_OBS) - 1):
        D = float(D_at(recipe, T_OBS[k]))
        f = lambda t, z: [mu_true(z[0], max(z[1], 0)) * z[0] - D * z[0],
                          -mu_true(z[0], max(z[1], 0)) * z[0] / Y + D * (S_IN - z[1])]
        sol = solve_ivp(f, (T_OBS[k], T_OBS[k + 1]), x, rtol=1e-8, atol=1e-10)
        x = list(sol.y[:, -1])
        out.append(list(x))
    y = np.array(out)
    if noise:
        rng = rng or np.random.default_rng()
        y = y + SIGMA * rng.standard_normal(y.shape)
    return y


def to_unit(x):
    return (np.asarray(x) - LB) / (UB - LB)


def from_unit(u):
    return LB + np.clip(u, 0, 1) * (UB - LB)
