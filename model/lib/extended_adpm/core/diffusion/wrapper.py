__all__ = ["DTWrapper", "DTCTWrapper", "CTWrapper", "SplitDTWrapper"]

import numpy as np
import torch
import torch.nn as nn
import logging
from core.utils.compatible import Wrapper
import core.func as func


def _rescale_timesteps(n, N, flag):
    if flag:
        return n * 1000.0 / float(N)
    return n


class DTWrapper(Wrapper):  # for compatible in models with discrete timesteps (DT)
    r"""
        The forward process is q(x_0, x_1, ..., x_N), which is indexed from 0 to N
        Some codes use different indexes, such as q(x_-1, x_0, ..., x_N-1)
    """
    def __init__(self, model: nn.Module, typ: str, rescale_timesteps: bool, shift1: bool, N, bipartition=None):
        r"""
        Args:
            shift1: whether to shift the index
        """
        super().__init__(model)
        self.typ = typ
        self.rescale_timesteps = rescale_timesteps
        self.shift1 = shift1
        self.N = N
        self.bipartition = bipartition if bipartition is not None else '_' in self.typ
        logging.info('DTWrapper with typ={}, rescale_timesteps={}, shift1={}, N={}, bipartition={}'
                     .format(typ, rescale_timesteps, shift1, N, bipartition))

    def __call__(self, xn, n):
        if self.shift1:
            n = n - 1
        if np.isscalar(n):
            n = np.array([n] * xn.size(0))
        n = torch.tensor(n).to(xn)
        out = self.model(xn, _rescale_timesteps(n, self.N, self.rescale_timesteps))
        if self.bipartition:
            out1, out2 = func.bipartition(out)
            return out1, out2
        else:
            return out


class SplitDTWrapper(Wrapper):
    def __init__(self, model0: nn.Module, model1: nn.Module, split_idx: int, typ: str, rescale_timesteps: bool,
                 shift1: bool, N, bipartition=None):
        super().__init__(None)
        self.model0 = DTWrapper(model0, typ=typ, rescale_timesteps=rescale_timesteps,
                                shift1=shift1, N=N, bipartition=bipartition)
        self.model1 = DTWrapper(model1, typ=typ, rescale_timesteps=rescale_timesteps,
                                shift1=shift1, N=N, bipartition=bipartition)
        self.typ = typ
        assert typ == 'eps'
        self.split_idx = split_idx

    def __call__(self, xn, n):
        assert np.isscalar(n)
        if n <= self.split_idx:
            return self.model0(xn, n)
        else:
            return self.model1(xn, n)


class CT2DTWrapper(Wrapper):  # convert a continuous time model to discrete time model
    r"""
        The forward process is q(x_[0,T])
        n -> t = n * T / N
        especially,
        n=0 -> t=0, data
        n=N -> t=T
    """
    def __init__(self, model: nn.Module, typ: str, N, T=1, bipartition=None):
        super().__init__(model)
        self.typ = typ
        self.T = T
        self.N = N
        self.bipartition = bipartition or '_' in self.typ
        logging.info('DTCTWrapper with typ={}, T={}, N={}, bipartition={}'.format(typ, T, N, bipartition))

    def __call__(self, xn, n):
        if np.isscalar(n):
            n = np.array([n] * xn.size(0))
        n = torch.tensor(n).to(xn)
        t = n / self.N
        out = self.model(xn, t * 999)  # follow SDE
        if self.bipartition:
            out1, out2 = func.bipartition(out)
            return out1, out2
        else:
            return out


class CTWrapper(Wrapper):  # continuous time model
    r"""
        The forward process is q(x_[0,T])
    """
    def __init__(self, model: nn.Module, typ: str, T=1, bipartition=None):
        super().__init__(model)
        self.typ = typ
        self.T = T
        self.bipartition = bipartition or '_' in self.typ
        logging.info('CTWrapper with typ={}, T={}, bipartition={}'.format(typ, T, bipartition))

    def __call__(self, xt, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        t = t.to(xt.device)
        if t.dim() == 0:
            t = func.duplicate(t, xt.size(0))
        out = self.model(xt, t * 999)  # follow SDE
        if self.bipartition:
            out1, out2 = func.bipartition(out)
            return out1, out2
        else:
            return out
