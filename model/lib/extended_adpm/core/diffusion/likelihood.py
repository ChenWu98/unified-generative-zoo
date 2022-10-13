import core.func as func
import torch
from .trajectory import _choice_steps
import logging
from .dtdpm import DDPM
from .utils import report_statistics
import numpy as np
from core.evaluate.score import score_on_dataset
from .sde import ReverseSDE, ODE
from scipy import integrate


@ torch.no_grad()
def nelbo_dtdpm(dtdpm, x0, rev_var_type, trajectory='linear', sample_steps=None, ms_eps=None, nll_terms=None):
    assert isinstance(dtdpm, DDPM)
    N = dtdpm.N
    sample_steps = sample_steps or N
    ns = _choice_steps(N, sample_steps, trajectory, ms_eps=ms_eps, nll_terms=nll_terms, betas=dtdpm.betas)
    timesteps = [0] + ns
    logging.info("nelbo_dtdpm with rev_var_type={}, trajectory={}, sample_steps={}"
                 .format(rev_var_type, trajectory, sample_steps))
    return _nelbo_dtdpm(dtdpm, x0, rev_var_type, timesteps, ms_eps)


@ torch.no_grad()
def _nelbo_dtdpm(dtdpm, x0, rev_var_type, timesteps, ms_eps=None):
    assert isinstance(dtdpm, DDPM)
    nelbo = torch.zeros(x0.size(0), device=x0.device)
    rev_terms = []

    mu_q = dtdpm.cum_alphas[timesteps[-1]] ** 0.5 * x0
    var_q = dtdpm.cum_betas[timesteps[-1]]
    mu_p = torch.zeros_like(mu_q)
    var_p = 1.
    term = func.kl_between_normal(mu_q, var_q, mu_p, var_p).flatten(1).sum(1)
    nelbo += term
    rev_terms.append(term)

    for s, t in list(zip(timesteps, timesteps[1:]))[::-1]:
        dtdpm.statistics = {}
        eps = torch.randn_like(x0)
        xt = dtdpm.cum_alphas[t] ** 0.5 * x0 + dtdpm.cum_betas[t] ** 0.5 * eps

        mu_p, var_p = dtdpm.predict_xprev_cov_xprev(xt, s, t, rev_var_type, ms_eps)
        if s == 0:
            var_p = dtdpm.tilde_beta(1, 2)
        dtdpm.statistics['var_p'] = var_p.mean().item()

        if s != 0:
            mu_q = dtdpm.q_posterior_mean(x0, s, t, xt=xt)
            var_q = dtdpm.tilde_beta(s, t)
            dtdpm.statistics['var_q'] = var_q.item()
            term = func.kl_between_normal(mu_q, var_q, mu_p, var_p).flatten(1).sum(1)
        else:
            term = -func.log_discretized_normal(x0, mu_p, var_p).flatten(1).sum(1)
        nelbo += term
        rev_terms.append(term)
        report_statistics(s, t, dtdpm.statistics)

    return nelbo, rev_terms[::-1]


@ torch.no_grad()
def get_nelbo_terms(dtdpm, dataset, batch_size, rev_var_type):
    assert isinstance(dtdpm, DDPM)
    N = dtdpm.N

    logging.info("get_nelbo_terms with rev_var_type={}".format(rev_var_type))

    F = np.full((N + 1, N + 1), float('inf'))  # F[s, t] with 0 <= s < t <= N

    for t in range(1, N + 1):
        def fn(x0):
            d = int(np.prod(x0.shape[1:]))
            eps = torch.randn_like(x0)
            xt = dtdpm.cum_alphas[t] ** 0.5 * x0 + dtdpm.cum_betas[t] ** 0.5 * eps

            if rev_var_type == 'optimal':
                x0_pred, eps_pred, cov_x0_pred = dtdpm.predict_x0_eps_cov_x0(xt, t)
            else:
                x0_pred, eps_pred = dtdpm.predict_x0_eps(xt, t)
                cov_x0_pred = None
            x0_pred_err = (x0 - x0_pred).pow(2)

            tmp = []

            mu_p = x0_pred
            var_p = dtdpm.tilde_beta(1, 2) if t == 1 else dtdpm.cum_betas[t]  # fix the last step variance
            nll = -func.log_discretized_normal(x0, mu_p, var_p).flatten(1).sum(1)  # -E_q log p(x0|xt)
            tmp.append(nll)

            for s in range(1, t):
                sigma2 = dtdpm._predict_cov_prev(s, t, rev_var_type, cov_x0_pred)
                square_term = (x0_pred_err / sigma2).flatten(1).sum(1)
                coeff = 0.5 * dtdpm.cum_alphas[s] * dtdpm.skip_betas[s, t] ** 2 / dtdpm.cum_betas[t] ** 2

                tilde_beta = dtdpm.tilde_beta(s, t)
                ratio = tilde_beta / sigma2

                c = 0.5 * (ratio - ratio.log() - 1.)
                if c.dim() > 1:
                    c = func.mean_flat(c)
                c = c * d

                tmp.append(c + coeff * square_term)

            return tuple(tmp)
        F[0: t, t] = score_on_dataset(dataset, fn, batch_size)
        logging.info("nll[{}]={}".format(t, F[0, t]))

    def last_term_fn(x0):
        mu_q = dtdpm.cum_alphas[N] ** 0.5 * x0
        var_q = dtdpm.cum_betas[N]
        mu_p = torch.zeros_like(mu_q)
        var_p = 1.
        term = func.kl_between_normal(mu_q, var_q, mu_p, var_p).flatten(1).sum(1)
        return term
    last_term = score_on_dataset(dataset, last_term_fn, batch_size)

    res = {"F": F, "last_term": last_term}
    return res


@torch.no_grad()
def ode_nll(ode, x, t_init=1e-5, hutchinson_type='Rademacher', rtol=1e-5, atol=1e-5, method='RK45'):
    r"""
    Calculate -log p_{t_init} (x) of the score ODE
    See `Score-Based Generative Modeling through Stochastic Differential Equations`
    Here we calculate the drift of the ODE and its divergence together for a better efficiency
    Note that p_{t_init} is a continuous distribution, which does not directly correspond to bpd
    """
    assert isinstance(ode, ODE)

    shape = x.shape
    B = shape[0]  # batch size

    if hutchinson_type == 'Gaussian':
        z = torch.randn_like(x)
    elif hutchinson_type == 'Rademacher':
        z = torch.randint_like(x, low=0, high=2).float() * 2 - 1.
    else:
        raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")

    def to_flattened_numpy(_x):
        """Flatten a torch tensor `x` and convert it to numpy."""
        return _x.detach().cpu().numpy().reshape((-1,))

    def from_flattened_numpy(_x, _shape):
        """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
        return torch.from_numpy(_x.reshape(_shape))

    def ode_func(t, state):  # state = (xt, -log p_t(xt)), represented by a numpy array
        xt = from_flattened_numpy(state[:-B], shape).to(x)
        t = torch.tensor(t, device=x.device)
        with torch.enable_grad():
            xt.requires_grad_(True)
            drift = ode.drift(xt, t)
            drift_z = (drift * z).sum()
            grad_drift_z = torch.autograd.grad(drift_z, xt)[0]
        xt.requires_grad_(False)
        div = func.inner_product(grad_drift_z, z)
        drift_numpy = to_flattened_numpy(drift)
        div_numpy = to_flattened_numpy(div)
        return np.concatenate([drift_numpy, div_numpy], axis=0)

    init = np.concatenate([to_flattened_numpy(x), np.zeros((B,))], axis=0)
    solution = integrate.solve_ivp(ode_func, (t_init, 1), init, rtol=rtol, atol=atol, method=method)
    nfe = solution.nfev  # number of function evaluations
    zp = solution.y[:, -1]
    x1 = from_flattened_numpy(zp[:-B], shape).to(x)
    delta_logp = from_flattened_numpy(zp[-B:], (B,)).to(x)  # the integral of drift divergence from t_init to 1
    prior_logp = func.log_normal(x1, 0, 1).flatten(1).sum(1)  # the standard normal distribution
    nll = -(prior_logp + delta_logp)
    return nll, prior_logp, delta_logp, nfe
