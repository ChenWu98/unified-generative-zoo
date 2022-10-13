__all__ = ["DTDSM", "DTDSDM", "DTDSDMErr", "CTDSDM", "CTDSDMErr", "CTDSM"]


import torch
from .base import NaiveCriterion
import core.func as func
import logging


def dt_dsm(x0, wrapper, schedule):
    n, eps, xn = schedule.sample(x0)
    eps_pred = wrapper(xn, n)
    return func.sos(eps - eps_pred)


def dt_dsdm(x0, wrapper, schedule):
    n, eps, xn = schedule.sample(x0)
    if wrapper.typ == 'eps_eps2':
        eps_pred, eps2_pred = wrapper(xn, n)
    elif wrapper.typ == 'eps_epsc':
        eps_pred, epsc_pred = wrapper(xn, n)
        eps2_pred = epsc_pred + eps_pred.pow(2)
    else:
        raise NotImplementedError
    return func.sos(eps - eps_pred), func.sos(eps.pow(2) - eps2_pred)


def dt_dsdm_err(x0, wrapper, schedule):
    n, eps, xn = schedule.sample(x0)
    eps_pred, epsc_pred = wrapper(xn, n)
    eps_err = eps - eps_pred
    return func.sos(eps_err), func.sos(eps_err.detach().pow(2) - epsc_pred)


def ct_dsdm(x0, wrapper, sde, t_init=1e-5):
    t, eps, xt = sde.sample(x0, t_init)
    if wrapper.typ == 'eps_eps2':
        eps_pred, eps2_pred = wrapper(xt, t)
    elif wrapper.typ == 'eps_epsc':
        eps_pred, epsc_pred = wrapper(xt, t)
        eps2_pred = epsc_pred + eps_pred.pow(2)
    else:
        raise NotImplementedError
    return func.sos(eps - eps_pred), func.sos(eps.pow(2) - eps2_pred)


def ct_dsm(x0, wrapper, sde, t_init=1e-5):
    t, eps, xt = sde.sample(x0, t_init)
    eps_pred = wrapper(xt, t)
    return func.mos(eps - eps_pred)


def ct_dsdm_err(x0, wrapper, sde, t_init=1e-5):
    t, eps, xt = sde.sample(x0, t_init)
    eps_pred, epsc_pred = wrapper(xt, t)
    eps_err = eps - eps_pred
    return func.sos(eps_err), func.sos(eps_err.detach().pow(2) - epsc_pred)


class DTDSM(NaiveCriterion):
    def __init__(self, schedule, wrapper, **kwargs):
        assert wrapper.typ == "eps"
        super().__init__(wrapper, **kwargs)
        self.schedule = schedule

    def objective(self, v, **kwargs):
        return dt_dsm(v, self.wrapper, self.schedule)


class DTDSDM(NaiveCriterion):
    def __init__(self, schedule, ratio, wrapper, **kwargs):
        assert wrapper.typ in ["eps_eps2", "eps_epsc"]
        super().__init__(wrapper, **kwargs)
        self.schedule = schedule
        self.ratio = ratio
        logging.info(f'DTDSDM with ratio={self.ratio} wrapper.typ={wrapper.typ}')

    def objective(self, v, **kwargs):
        dsm_obj, ddm_obj = dt_dsdm(v, self.wrapper, self.schedule)
        with torch.no_grad():
            self.statistics["dsm_obj"] = dsm_obj_mean = dsm_obj.mean().item()
            self.statistics["ddm_obj"] = ddm_obj_mean = ddm_obj.mean().item()
            self.statistics["ratio"] = ratio = dsm_obj_mean / ddm_obj_mean if self.ratio == "adaptive" else self.ratio
        return dsm_obj + ratio * ddm_obj


class DTDSDMErr(NaiveCriterion):
    def __init__(self, schedule, ratio, wrapper, **kwargs):
        assert wrapper.typ in ["eps_epsc"]
        super().__init__(wrapper, **kwargs)
        self.schedule = schedule
        self.ratio = ratio
        logging.info(f'DTDSDMErr with ratio={self.ratio} wrapper.typ={wrapper.typ}')

    def objective(self, v, **kwargs):
        dsm_obj, ddm_obj = dt_dsdm_err(v, self.wrapper, self.schedule)
        with torch.no_grad():
            self.statistics["dsm_obj"] = dsm_obj_mean = dsm_obj.mean().item()
            self.statistics["ddm_obj"] = ddm_obj_mean = ddm_obj.mean().item()
            self.statistics["ratio"] = ratio = dsm_obj_mean / ddm_obj_mean if self.ratio == "adaptive" else self.ratio
        return dsm_obj + ratio * ddm_obj


class CTDSM(NaiveCriterion):
    def __init__(self, sde, wrapper, **kwargs):
        assert wrapper.typ == 'eps'
        super().__init__(wrapper, **kwargs)
        self.sde = sde

    def objective(self, v, **kwargs):
        dsm_obj = ct_dsm(v, self.wrapper, self.sde)
        return dsm_obj


class CTDSDM(NaiveCriterion):
    def __init__(self, sde, ratio, wrapper, **kwargs):
        assert wrapper.typ in ["eps_eps2", "eps_epsc"]
        super().__init__(wrapper, **kwargs)
        self.sde = sde
        self.ratio = ratio
        logging.info(f'CTDSDM with ratio={self.ratio} wrapper.typ={wrapper.typ} sde={type(self.sde)}')

    def objective(self, v, **kwargs):
        dsm_obj, ddm_obj = ct_dsdm(v, self.wrapper, self.sde)
        with torch.no_grad():
            self.statistics["dsm_obj"] = dsm_obj_mean = dsm_obj.mean().item()
            self.statistics["ddm_obj"] = ddm_obj_mean = ddm_obj.mean().item()
            self.statistics["ratio"] = ratio = dsm_obj_mean / ddm_obj_mean if self.ratio == "adaptive" else self.ratio
        return dsm_obj + ratio * ddm_obj


class CTDSDMErr(NaiveCriterion):
    def __init__(self, sde, ratio, wrapper, **kwargs):
        assert wrapper.typ in ["eps_epsc"]
        super().__init__(wrapper, **kwargs)
        self.sde = sde
        self.ratio = ratio
        logging.info(f"CTDSDMErr with ratio={self.ratio} wrapper.typ={wrapper.typ} sde={type(self.sde)}")

    def objective(self, v, **kwargs):
        dsm_obj, ddm_obj = ct_dsdm_err(v, self.wrapper, self.sde)
        with torch.no_grad():
            self.statistics["dsm_obj"] = dsm_obj_mean = dsm_obj.mean().item()
            self.statistics["ddm_obj"] = ddm_obj_mean = ddm_obj.mean().item()
            self.statistics["ratio"] = ratio = dsm_obj_mean / ddm_obj_mean if self.ratio == "adaptive" else self.ratio
        return dsm_obj + ratio * ddm_obj
