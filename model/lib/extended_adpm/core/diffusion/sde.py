import torch
import numpy as np
import math
import core.func as func
from .schedule import Schedule


class SDE(object):
    r"""
        dx = f(x, t)dt + g(t) dw with 0 <= t <= 1
        f(x, t) is the drift
        g(t) is the diffusion
    """
    def drift(self, x, t):
        raise NotImplementedError

    def diffusion(self, t):
        raise NotImplementedError

    def cum_beta(self, t):  # the variance of xt|x0
        raise NotImplementedError

    def marginal_prob(self, x0, t):  # the mean and std of q(xt|x0)
        raise NotImplementedError

    def sample(self, x0, t_init):  # sample from q(xn|x0), where n is uniform
        t = torch.rand(x0.shape[0], device=x0.device) * (1. - t_init) + t_init
        mean, std = self.marginal_prob(x0, t)
        eps = torch.randn_like(x0)
        xt = mean + func.stp(std, eps)
        return t, eps, xt


class ReverseSDE(object):
    r"""
        dx = [f(x, t) - g(t)^2 s(x, t)] dt + g(t) dw
    """
    def __init__(self, sde, wrapper):
        self.sde = sde  # the forward sde
        self.wrapper = wrapper
        if self.wrapper.typ == 'eps':
            def score(x, t):
                cum_beta = self.sde.cum_beta(t)
                eps_pred = self.wrapper(x, t)
                return func.stp(-cum_beta.rsqrt(), eps_pred)
        else:
            raise NotImplementedError
        self.score = score

    def drift(self, x, t):
        drift = self.sde.drift(x, t)  # f(x, t)
        diffusion = self.sde.diffusion(t)  # g(t)
        score = self.score(x, t)
        return drift - func.stp(diffusion ** 2, score)

    def diffusion(self, t):
        return self.sde.diffusion(t)


class ODE(object):
    r"""
        dx = [f(x, t) - g(t)^2 s(x, t)] dt
    """

    def __init__(self, sde, wrapper):
        self.sde = sde  # the forward sde
        self.wrapper = wrapper
        if self.wrapper.typ == 'eps':
            def score(x, t):
                cum_beta = self.sde.cum_beta(t)
                eps_pred = self.wrapper(x, t)
                return func.stp(-cum_beta.rsqrt(), eps_pred)
        else:
            raise NotImplementedError
        self.score = score

    def drift(self, x, t):
        drift = self.sde.drift(x, t)  # f(x, t)
        diffusion = self.sde.diffusion(t)  # g(t)
        score = self.score(x, t)
        return drift - 0.5 * func.stp(diffusion ** 2, score)

    def diffusion(self, t):
        return 0


class VPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20):
        # 0 <= t <= 1
        self.beta_0 = beta_min
        self.beta_1 = beta_max

    def drift(self, x, t):
        return -0.5 * func.stp(self.squared_diffusion(t), x)

    def diffusion(self, t):
        return self.squared_diffusion(t) ** 0.5

    def squared_diffusion(self, t):  # beta(t)
        return self.beta_0 + t * (self.beta_1 - self.beta_0)

    def squared_diffusion_integral(self, s, t):  # \int_s^t beta(tau) d tau
        return self.beta_0 * (t - s) + (self.beta_1 - self.beta_0) * (t ** 2 - s ** 2) * 0.5

    def skip_beta(self, s, t):  # beta_{t|s}, Cov[xt|xs]=beta_{t|s} I
        return 1. - self.skip_alpha(s, t)

    def skip_alpha(self, s, t):  # alpha_{t|s}, E[xt|xs]=alpha_{t|s}**0.5 xs
        x = -self.squared_diffusion_integral(s, t)
        if isinstance(x, torch.Tensor):
            return x.exp()
        elif isinstance(x, np.ndarray):
            return np.exp(x)
        else:
            return math.exp(x)

    def cum_beta(self, t):
        return self.skip_beta(0, t)

    def cum_alpha(self, t):
        return self.skip_alpha(0, t)

    def marginal_prob(self, x0, t):  # the mean and std of q(xt|x0)
        alpha = self.cum_alpha(t)
        beta = self.cum_beta(t)
        mean = func.stp(alpha ** 0.5, x0)  # E[xt|x0]
        std = beta ** 0.5  # Cov[xt|x0] ** 0.5
        return mean, std

    def get_schedule(self, N):
        ts = np.linspace(0, 1, N + 1, dtype=np.float64)
        betas = [0.]
        for s, t in zip(ts, ts[1:]):
            betas.append(self.skip_beta(s, t))
        betas = np.array(betas, dtype=np.float64)
        return Schedule(betas=betas)


class SubVPSDE(object):
    def __init__(self, beta_min=0.1, beta_max=20):
        self.beta_0 = beta_min
        self.beta_1 = beta_max

    def marginal_prob(self, x0, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = func.stp(log_mean_coeff.exp(), x0)
        std = 1. - (2. * log_mean_coeff).exp()
        return mean, std
