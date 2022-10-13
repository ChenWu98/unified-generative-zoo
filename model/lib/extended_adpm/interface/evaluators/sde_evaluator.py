import os
from core.evaluate import grid_sample, sample2dir
from .base import Evaluator
from core.diffusion.sample import euler_maruyama
from core.diffusion.likelihood import ode_nll
from core.diffusion.sde import ReverseSDE, ODE
from core.evaluate.score import score_on_dataset, cat_score_on_dataset
from core.diffusion.utils import statistics2str
from interface.utils.interact import Interact
from interface.datasets import DatasetFactory
import torch
from core.utils import global_device
from torch.utils.data import Subset
import numpy as np
import logging
import math
import core.func as func
import random


class SDEEvaluator(Evaluator):
    def __init__(self, wrapper, options: dict,
                 dataset: DatasetFactory = None, interact: Interact = None):
        r""" Evaluate DPM with continuous timesteps (i.e., SDE)
        Args:
            wrapper: an object of Wrapper
            options: a dict, evaluation function name -> arguments of the function
                Example: {"grid_sample": {"nrow": 10, "ncol": 10}}
            dataset: an instance of DatasetFactory
            interact: an instance of Interact
        """
        super().__init__(options)
        self.wrapper = wrapper
        self.dataset = dataset
        self.unpreprocess_fn = None if self.dataset is None else self.dataset.unpreprocess
        self.interact = interact

    def grid_sample(self, it, sde, path, sample_steps=None, nrow=10, ncol=10):
        fname = os.path.join(path, "%d.png" % it)

        rsde = ODE(sde, self.wrapper)

        def sample_fn(n_samples):
            x_init = torch.randn(n_samples, *self.dataset.data_shape, device=global_device())
            return euler_maruyama(rsde, x_init, sample_steps=sample_steps)
        grid_sample(fname, nrow, ncol, sample_fn, self.unpreprocess_fn)

    def sample2dir(self, path, n_samples, batch_size, sde, reverse_type, sample_steps=None, persist=True, it=None):
        os.makedirs(path, exist_ok=True)

        logging.info(f"sample2dir with {n_samples} samples, reverse_type={reverse_type}")
        if self.dataset.fid_stat is not None:
            assert os.path.exists(self.dataset.fid_stat)

        if reverse_type == "sde":
            rsde = ReverseSDE(sde, self.wrapper)
        elif reverse_type == "ode":
            rsde = ODE(sde, self.wrapper)
        else:
            raise NotImplementedError

        def sample_fn(_n_samples):
            x_init = torch.randn(_n_samples, *self.dataset.data_shape, device=global_device())
            return euler_maruyama(rsde, x_init, sample_steps=sample_steps)
        sample2dir(path, n_samples, batch_size, sample_fn, self.unpreprocess_fn, persist)
        if self.dataset.fid_stat is not None:
            from tools.fid_score import calculate_fid_given_paths
            fid = calculate_fid_given_paths((self.dataset.fid_stat, path))
            logging.info("fid={}".format(fid))

    def nll(self, fname, batch_size, sde, reverse_type, n_samples=None, partition="test", t_init=1e-5,
            it=None):
        dataset = self.dataset.get_data(partition, labelled=False)
        n_samples = n_samples or len(dataset)
        idxes = random.sample(list(range(len(dataset))), n_samples)
        dataset = Subset(dataset, idxes)

        logging.info(f'nll with {n_samples} {partition} samples, reverse_type={reverse_type}, t_init={t_init}')

        if reverse_type == 'ode':
            rsde = ODE(sde, self.wrapper)

            def score_fn(x):
                r"""
                x is discrete data scaled from {0, ..., 255} to [-1, 1]

                Calculate the bpd of the discrete distribution P(x) formed by:
                x=floor(128 x_{t_init} + 128), where x_{t_init} ~ p_{t_init}(x_{t_init})

                We have -log2 P(x) <= -E_u log2 p_{t_init}( (x+u-128)/128 ) + 7D
                """
                x_discrete = (x + 1) * 0.5 * 255  # {0, ..., 255}
                x_u = x_discrete + torch.rand_like(x_discrete)  # uniform dequantization
                x_scale = (x_u - 128) / 128
                _nll, _prior_logp, _delta_logp, nfe = ode_nll(rsde, x_scale, t_init=t_init)  # -log p_{init}((x+u-128)/128)
                _bpd = _nll / (self.dataset.data_dim * math.log(2.)) + 7
                statistics = dict(bs=x.size(0), bpd=_bpd.mean().item(), nll=_nll.mean().item(),
                                  prior_logp=_prior_logp.mean().item(), delta_logp=_delta_logp.mean().item(), nfe=nfe)
                logging.info(statistics2str(statistics))
                return _bpd, _nll, _prior_logp, _delta_logp

            bpd, nll, prior_logp, delta_logp = score_on_dataset(dataset, score_fn, batch_size)  # averaged over dataset

            self.interact.report_scalar(bpd, it, 'bpd')
            self.interact.report_scalar(nll, it, 'nll')
            self.interact.report_scalar(prior_logp, it, 'prior_logp')
            self.interact.report_scalar(delta_logp, it, 'delta_logp')
            torch.save(dict(bpd=bpd, nll=nll, prior_logp=prior_logp, delta_logp=delta_logp), fname)

        else:
            raise NotImplementedError
