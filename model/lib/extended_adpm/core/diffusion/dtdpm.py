__all__ = ["DTDPM", "DDPM", "DDIM"]

import numpy as np
import logging
import core.func as func
from core.utils import global_device
import torch


class DTDPM(object):  # diffusion with discrete timesteps
    r"""
        E[xs|xt] = E[ E[xs|xt,x0] |xt] = E[xs|xt,x0=E[x0|xt]] in DDPM or DDIM forward process
    """
    def __init__(self, wrapper, schedule, clip_x0: bool, clip_cov_x0=None, avg_cov=False):
        assert wrapper.typ in ['eps', 'x0', 'eps_eps2', 'eps_epsc', 'eps_iddpm']
        self.wrapper = wrapper

        self.schedule = schedule
        self.N = schedule.N
        self.betas = schedule.betas
        self.alphas = schedule.alphas
        self.cum_betas = schedule.cum_betas
        self.cum_alphas = schedule.cum_alphas
        self.skip_alphas = schedule.skip_alphas
        self.skip_betas = schedule.skip_betas
        self.tilde_beta = schedule.tilde_beta

        self.clip_x0 = clip_x0
        self.clip_cov_x0 = clip_x0 if clip_cov_x0 is None else clip_cov_x0
        self.avg_cov = avg_cov

        logging.info("DTDPM with clip_x0={} clip_cov_x0={} avg_cov={}".format(self.clip_x0, self.clip_cov_x0, self.avg_cov))
        self.statistics = {}

    def predict_x0_eps(self, xt, t):  # estimate E[x0|xt], E[eps|xt] w.r.t. q
        if self.wrapper.typ in ['eps', 'x0']:
            pred = self.wrapper(xt, t)
        elif self.wrapper.typ.startswith('eps_'):
            pred, _ = self.wrapper(xt, t)
        else:
            raise NotImplementedError
        x0_pred, eps_pred = self._predict_x0_eps(pred, xt, t)
        return x0_pred, eps_pred

    def _predict_x0_eps(self, pred, xt, t):  # pred: the direct output of the first model
        if self.wrapper.typ == 'eps' or self.wrapper.typ.startswith('eps_'):
            eps_pred = pred
            x0_pred = self.cum_alphas[t] ** -0.5 * xt - (self.cum_betas[t] / self.cum_alphas[t]) ** 0.5 * eps_pred
        elif self.wrapper.typ == 'x0':
            x0_pred = pred
            eps_pred = - (self.cum_alphas[t] / self.cum_betas[t]) ** 0.5 * x0_pred + (1. / self.cum_betas[t] ** 0.5) * xt
        else:
            raise NotImplementedError
        if self.clip_x0:
            x0_pred = x0_pred.clamp(-1., 1.)
        return x0_pred, eps_pred

    def predict_cov_x0(self, xt, t, ms_eps=None):  # estimate Cov[x0|xt], for typ = "optimal"
        if ms_eps is not None:
            return self._e_cov_x0(t, ms_eps)

        if self.wrapper.typ in ['eps_eps2', 'eps_epsc']:
            pred1, pred2 = self.wrapper(xt, t)
        elif self.wrapper.typ in ["eps"]:
            pred1 = self.wrapper(xt, t)
            pred2 = None
        else:
            raise NotImplementedError
        cov_x0_pred = self._predict_cov_x0(pred1, pred2, xt, t)
        return cov_x0_pred

    def _predict_cov_x0(self, pred1, pred2, xt, t):  # for typ = "optimal"
        if self.wrapper.typ == 'eps_eps2':
            eps_pred, eps2_pred = pred1, pred2
            cov_x0_pred = (self.cum_betas[t] / self.cum_alphas[t]) * (eps2_pred - eps_pred.pow(2))
        elif self.wrapper.typ == 'eps_epsc':
            eps_pred, epsc_pred = pred1, pred2
            cov_x0_pred = (self.cum_betas[t] / self.cum_alphas[t]) * epsc_pred
        elif self.wrapper.typ == 'eps' and pred2 is None:
            # a huristic estimate: cov[v|w] ≈ beta/alpha (1 - beta s(w)^2) = beta/alpha (1 - eps(w)^2)
            eps_pred = pred1
            cov_x0_pred = (self.cum_betas[t] / self.cum_alphas[t]) * (1 - eps_pred.pow(2))
        else:
            raise NotImplementedError
        if self.avg_cov:
            cov_x0_pred = func.mean_flat(cov_x0_pred, keepdim=True)
        self.statistics['cov_x0'] = cov_x0_pred.mean().item()
        if self.clip_cov_x0:
            cov_x0_pred = cov_x0_pred.clamp(0., 1.)
            self.statistics['cov_x0_clip'] = cov_x0_pred.mean().item()
        return cov_x0_pred

    def _e_cov_x0(self, t, ms_eps):  # estimate E Cov[x0|xt]
        cov_x0_pred = self.cum_betas[t] / self.cum_alphas[t] * (1. - ms_eps[t])
        if not isinstance(cov_x0_pred, torch.Tensor):
            cov_x0_pred = torch.tensor(cov_x0_pred, device=global_device())
        self.statistics['cov_x0'] = cov_x0_pred.item()
        if self.clip_cov_x0:
            cov_x0_pred = cov_x0_pred.clamp(0., 1.)
            self.statistics['cov_x0_clip'] = cov_x0_pred.item()
        return cov_x0_pred

    def predict_x0_eps_cov_x0(self, xt, t, ms_eps=None):  # for typ = "optimal"
        if ms_eps is not None:
            x0_pred, eps_pred = self.predict_x0_eps(xt, t)
            cov_x0_pred = self.predict_cov_x0(None, t, ms_eps)
            return x0_pred, eps_pred, cov_x0_pred

        if self.wrapper.typ in ['eps_eps2', 'eps_epsc']:
            pred1, pred2 = self.wrapper(xt, t)
        elif self.wrapper.typ in ["eps"]:
            pred1 = self.wrapper(xt, t)
            pred2 = None
        else:
            raise NotImplementedError
        x0_pred, eps_pred = self._predict_x0_eps(pred1, xt, t)
        cov_x0_pred = self._predict_cov_x0(pred1, pred2, xt, t)
        return x0_pred, eps_pred, cov_x0_pred

    def q_posterior_mean(self, x0, s, t, xt=None, eps=None):  # E[xs|xt,x0] w.r.t. q
        raise NotImplementedError

    def predict_xprev(self, xt, s, t):  # estimate E[xs|xt] w.r.t. q
        x0_pred, eps_pred = self.predict_x0_eps(xt, t)
        return self.q_posterior_mean(x0_pred, s, t, xt=xt, eps=eps_pred)

    def predict_cov_prev(self, xt, s, t, typ, ms_eps=None):  # estimate Cov[xs|xt] w.r.t. q
        cov_x0_pred = self.predict_cov_x0(xt, t, ms_eps) if typ == 'optimal' else None
        return self._predict_cov_prev(s, t, typ, cov_x0_pred)

    def _predict_cov_prev(self, s, t, typ, cov_x0_pred=None):
        raise NotImplementedError

    def predict_xprev_cov_xprev(self, xt, s, t, typ, ms_eps=None):
        if typ == 'optimal':
            x0_pred, eps_pred, cov_x0_pred = self.predict_x0_eps_cov_x0(xt, t, ms_eps)
        else:
            x0_pred, eps_pred = self.predict_x0_eps(xt, t)
            cov_x0_pred = None
        xprev_pred = self.q_posterior_mean(x0_pred, s, t, xt=xt, eps=eps_pred)
        sigma2 = self._predict_cov_prev(s, t, typ, cov_x0_pred)
        return xprev_pred, sigma2


class DDPM(DTDPM):
    def q_posterior_mean(self, x0, s, t, xt=None, eps=None):  # E[xs|xt,x0] w.r.t. q
        assert xt is not None
        coeff1 = self.skip_betas[s, t] * self.cum_alphas[s] ** 0.5 / self.cum_betas[t]
        coeff2 = self.skip_alphas[s, t] ** 0.5 * self.cum_betas[s] / self.cum_betas[t]
        return coeff1 * x0 + coeff2 * xt

    def _predict_cov_prev(self, s, t, typ, cov_x0_pred=None):
        sigma2_small = self.tilde_beta(s, t)
        self.statistics['sigma2_small'] = sigma2_small
        self.statistics['sigma2_big'] = self.skip_betas[s, t]
        if typ == 'small':
            sigma2 = sigma2_small
        elif typ == 'big':
            sigma2 = self.skip_betas[s, t]
        elif typ == 'optimal':
            coeff_cov_x0 = self.cum_alphas[s] * self.skip_betas[s, t] ** 2 / self.cum_betas[t] ** 2
            offset = coeff_cov_x0 * cov_x0_pred
            sigma2 = sigma2_small + offset
            self.statistics['coeff_cov_x0'] = coeff_cov_x0.item()
            self.statistics['offset'] = offset.mean().item()
        else:
            raise NotImplementedError
        if not isinstance(sigma2, torch.Tensor):
            sigma2 = torch.tensor(sigma2, device=global_device())
        return sigma2

    def predict_xprev_cov_xprev(self, xt, s, t, typ, ms_eps=None):
        if self.wrapper.typ == "eps_iddpm" and typ == "optimal" and ms_eps is None:
            eps_pred, model_var_values = self.wrapper(xt, t)
            x0_pred, eps_pred = self._predict_x0_eps(eps_pred, xt, t)
            xprev_pred = self.q_posterior_mean(x0_pred, s, t, xt=xt, eps=eps_pred)
            min_log = np.log(self.tilde_beta(s, t) if s > 0 else self.tilde_beta(1, 2))
            max_log = np.log(self.skip_betas[s, t])
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            sigma2 = model_log_variance.exp()
            return xprev_pred, sigma2
        else:
            return super().predict_xprev_cov_xprev(xt, s, t, typ, ms_eps)


class DDIM(DTDPM):
    def __init__(self, wrapper, schedule, clip_x0: bool, eta: float, clip_cov_x0=None, avg_cov=False):
        super().__init__(wrapper, schedule, clip_x0, clip_cov_x0=clip_cov_x0, avg_cov=avg_cov)
        self.eta = eta
        logging.info("DDIM with eta={}".format(eta))

    def q_posterior_mean(self, x0, s, t, xt=None, eps=None):  # E[xs|xt,x0] w.r.t. q
        # eps = (xt - self.cum_alphas[t] ** 0.5 * x0) / self.cum_betas[t] ** 0.5
        assert eps is not None
        sigma2_small = self.tilde_beta(s, t)
        lamb2 = self.eta ** 2 * sigma2_small

        coeff1 = self.cum_alphas[s] ** 0.5
        coeff2 = (self.cum_betas[s] - lamb2) ** 0.5
        return coeff1 * x0 + coeff2 * eps

    def _predict_cov_prev(self, s, t, typ, cov_x0_pred=None):
        sigma2_small = self.tilde_beta(s, t)
        lamb2 = self.eta ** 2 * sigma2_small
        if typ == 'small':
            sigma2 = lamb2
        elif typ == 'optimal':
            coeff_cov_x0 = (self.cum_alphas[s] ** 0.5 - ((self.cum_betas[s] - lamb2) * self.cum_alphas[t] / self.cum_betas[t]) ** 0.5) ** 2
            offset = coeff_cov_x0 * cov_x0_pred
            sigma2 = lamb2 + offset
        else:
            raise NotImplementedError
        if not isinstance(sigma2, torch.Tensor):
            sigma2 = torch.tensor(sigma2, device=global_device())
        return sigma2
