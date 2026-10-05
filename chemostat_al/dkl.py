"""DKL kinetics trained THROUGH THE ODE, as in the CHO study.

Network   in_dim -> 5 -> 5 -> 4, tanh, Xavier init           (CHO: 10 hidden)
GP        SVGP: CholeskyVariationalDistribution + whitened VariationalStrategy,
          learned inducing locations, ScaleKernel(RBF, ARD) on the features,
          M = 5 x recipes, initialised at the network image of observed inputs.

A GP realisation is represented by a whitened inducing draw v ~ q(v) = N(m, S):
    f(x) = mean(z) + K(z, Z) L^{-T} v,   z = net(x),  L = chol K(Z, Z)
v = m gives the posterior-mean function; a frozen random v gives one sampled
realisation used for a whole rollout (the CHO "cached sample").

Training (fit.py) uses these functions inside torchdiffeq rollouts; the
predictive for the acquisition uses the same whitened quantities.
"""
import torch, gpytorch
from torch import nn
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

torch.set_default_dtype(torch.float64)
HIDDEN, FEATURE_DIM, INDUCING_PER_RECIPE, JITTER = 5, 4, 5, 1e-5


class FeatureExtractor(nn.Sequential):
    def __init__(self, in_dim, generator=None):
        super().__init__(nn.Linear(in_dim, HIDDEN), nn.Tanh(),
                         nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
                         nn.Linear(HIDDEN, FEATURE_DIM))
        for layer in self:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, generator=generator)
                nn.init.zeros_(layer.bias)


class _SVGP(ApproximateGP):
    def __init__(self, inducing):
        q = CholeskyVariationalDistribution(inducing.shape[0])
        super().__init__(VariationalStrategy(self, inducing, q,
                                             learn_inducing_locations=True))
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=FEATURE_DIM))

    def forward(self, z):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(z), self.covar_module(z))


class DKL(nn.Module):
    """One scalar latent function f(x) on a box-normalised input."""

    def __init__(self, in_lb, in_ub, init_inputs, n_recipes, seed=0):
        super().__init__()
        self.register_buffer("lb", torch.as_tensor(in_lb))
        self.register_buffer("ub", torch.as_tensor(in_ub))
        gen = torch.Generator().manual_seed(seed)
        self.feature = FeatureExtractor(len(in_lb), gen)
        X = self._n(init_inputs)
        M = min(INDUCING_PER_RECIPE * n_recipes, len(X))
        idx = torch.randperm(len(X), generator=gen)[:M]
        with torch.no_grad():
            Z0 = self.feature(X[idx])
        self.gp = _SVGP(Z0.clone())
        self.M = M
        self._cache = {}                   # cached whitened inducing draw 'v'

    def _n(self, X):
        return (torch.as_tensor(X) - self.lb) / (self.ub - self.lb)

    # ---- fast evaluation path --------------------------------------------- #
    # Outside training the inducing points, kernel and variational parameters are
    # fixed, so chol K(Z, Z) and the variational factors are the same on every
    # call. Recomputing them -- through gpytorch's lazy-tensor kernel machinery --
    # made each call ~1 ms, i.e. ~60 ms per rollout. The cached path below
    # evaluates ScaleKernel(RBF-ARD) directly in torch and reuses the Cholesky.
    # It is exactly the same computation (verified to machine precision) and is
    # only used when gradients are off; training still goes through gpytorch.
    @staticmethod
    def _rbf(A, B, ls, scale):
        A, B = A / ls, B / ls
        d2 = (A * A).sum(-1)[:, None] + (B * B).sum(-1)[None, :] - 2.0 * A @ B.T
        return scale * torch.exp(-0.5 * d2.clamp_min(0.0))

    def _consts(self):
        key = tuple(p._version for p in self.parameters())
        pc = self.__dict__.get("_pc")
        if pc is not None and pc[0] == key:
            return pc[1]
        with torch.no_grad():
            q = self.gp.variational_strategy._variational_distribution
            Zi = self.gp.variational_strategy.inducing_points.detach().clone()
            ls = self.gp.covar_module.base_kernel.lengthscale.detach().clone()
            sc = self.gp.covar_module.outputscale.detach().clone()
            L = torch.linalg.cholesky(self._rbf(Zi, Zi, ls, sc)
                                      + JITTER * torch.eye(len(Zi)))
            c = dict(Zi=Zi, ls=ls, sc=sc, L=L,
                     m=q.variational_mean.detach().clone(),
                     LqT=torch.tril(q.chol_variational_covar).detach().T.clone(),
                     m0=self.gp.mean_module.constant.detach().clone())
        self.__dict__["_pc"] = (key, c)
        return c

    # ---- realisations (differentiable; used in training rollouts) -------- #
    def cache_sample(self, generator=None):
        q = self.gp.variational_strategy._variational_distribution
        m = q.variational_mean
        Lq = torch.tril(q.chol_variational_covar)
        e = torch.randn(m.shape, generator=generator)
        self._cache['v'] = m + Lq @ e

    def cache_mean(self):
        self._cache['v'] = self.gp.variational_strategy._variational_distribution.variational_mean

    def clear(self):
        self._cache = {}

    def realisation(self, x):
        if not torch.is_grad_enabled():
            c = self._consts()
            z = self.feature(self._n(x))
            alpha = torch.linalg.solve_triangular(c["L"].T, self._cache["v"].unsqueeze(-1),
                                                  upper=True)
            return c["m0"] + (self._rbf(z, c["Zi"], c["ls"], c["sc"]) @ alpha).squeeze(-1)
        Zi = self.gp.variational_strategy.inducing_points
        z = self.feature(self._n(x))
        Kzz = self.gp.covar_module(Zi).to_dense() + JITTER * torch.eye(len(Zi))
        L = torch.linalg.cholesky(Kzz)
        alpha = torch.linalg.solve_triangular(L.T, self._cache['v'].unsqueeze(-1), upper=True)
        Kxz = self.gp.covar_module(z, Zi).to_dense()
        return self.gp.mean_module.constant + (Kxz @ alpha).squeeze(-1)

    def kl(self):
        return self.gp.variational_strategy.kl_divergence().sum()

    # ---- predictive (acquisition, constraint, figures) ------------------- #
    @torch.no_grad()
    def predict(self, x):
        """Whitened SVGP predictive, computed from the same quantities as the
        realisations (so prediction and training are exactly consistent):
            a = L^{-1} k(Z, x)
            mean = m0 + a^T m,     var = k(x, x) - a^T a + a^T S a,  S = Lq Lq^T
        """
        c = self._consts()
        z = self.feature(self._n(x))
        a = torch.linalg.solve_triangular(c["L"], self._rbf(c["Zi"], z, c["ls"], c["sc"]),
                                          upper=False)
        mean = c["m0"] + a.T @ c["m"]
        var = c["sc"] - (a * a).sum(0) + ((c["LqT"] @ a) ** 2).sum(0)
        return mean, var.clamp_min(1e-12).sqrt()
