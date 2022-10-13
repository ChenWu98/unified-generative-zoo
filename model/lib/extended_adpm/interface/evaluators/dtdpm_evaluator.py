import os
from core.evaluate import grid_sample, sample2dir
from .base import Evaluator
from core.diffusion.sample import sample_dtdpm
from core.diffusion.dtdpm import DDPM, DDIM, DTDPM
from core.diffusion.likelihood import nelbo_dtdpm, get_nelbo_terms
from core.evaluate.score import score_on_dataset, cat_score_on_dataset
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


class DTDPMEvaluator(Evaluator):
    def __init__(self, wrapper, options: dict,
                 dataset: DatasetFactory = None, interact: Interact = None):
        r""" Evaluate DPM with discrete timesteps
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

    def grid_sample(self, it, schedule, rev_var_type, path, clip_x0=True, sample_steps=None, nrow=10, ncol=10):
        fname = os.path.join(path, "%d.png" % it)

        diffusion = DDPM(self.wrapper, schedule, clip_x0=clip_x0)

        def sample_fn(n_samples):
            x_init = torch.randn(n_samples, *self.dataset.data_shape, device=global_device())
            return sample_dtdpm(diffusion, x_init, rev_var_type, sample_steps=sample_steps)
        grid_sample(fname, nrow, ncol, sample_fn, self.unpreprocess_fn)

    def sample2dir(self, path, n_samples, batch_size, schedule, forward_type, rev_var_type, clip_x0=True, avg_cov=False,
                   trajectory='linear', sample_steps=None, clip_sigma_idx=0, clip_pixel=2, eta=None,
                   ms_eps_path=None, persist=True, it=None):
        os.makedirs(path, exist_ok=True)

        logging.info("sample2dir with {} samples".format(n_samples))
        if self.dataset.fid_stat is not None:
            assert os.path.exists(self.dataset.fid_stat)

        ms_eps = None
        if ms_eps_path is not None:
            logging.info("load ms_eps from {}".format(ms_eps_path))
            ms_eps = torch.load(ms_eps_path)

        if forward_type == "ddpm":
            diffusion = DDPM(self.wrapper, schedule, clip_x0=clip_x0, avg_cov=avg_cov)
        elif forward_type == "ddim":
            diffusion = DDIM(self.wrapper, schedule, clip_x0=clip_x0, eta=eta, avg_cov=avg_cov)
        else:
            raise NotImplementedError

        def sample_fn(_n_samples):
            x_init = torch.randn(_n_samples, *self.dataset.data_shape, device=global_device())
            return sample_dtdpm(diffusion, x_init, rev_var_type, trajectory=trajectory, sample_steps=sample_steps,
                                clip_sigma_idx=clip_sigma_idx, clip_pixel=clip_pixel, ms_eps=ms_eps)
        sample2dir(path, n_samples, batch_size, sample_fn, self.unpreprocess_fn, persist)
        if self.dataset.fid_stat is not None:
            from tools.fid_score import calculate_fid_given_paths
            fid = calculate_fid_given_paths((self.dataset.fid_stat, path))
            logging.info("fid={}".format(fid))

    def nll(self, fname, batch_size, schedule, rev_var_type, clip_x0=True, avg_cov=False,
            trajectory='linear', sample_steps=None, n_samples=None, partition="test",
            ms_eps_path=None, nll_terms_path=None, it=None):
        dataset = self.dataset.get_data(partition, labelled=False)
        n_samples = n_samples or len(dataset)
        idxes = random.sample(list(range(len(dataset))), n_samples)
        dataset = Subset(dataset, idxes)

        logging.info("nll with {} {} samples".format(n_samples, partition))

        ms_eps = None
        if ms_eps_path is not None:
            logging.info("load ms_eps from {}".format(ms_eps_path))
            ms_eps = torch.load(ms_eps_path)

        nll_terms = None
        if nll_terms_path is not None:
            logging.info("load nll_terms from {}".format(nll_terms_path))
            nll_terms = torch.load(nll_terms_path)

        diffusion = DDPM(self.wrapper, schedule, clip_x0=clip_x0, avg_cov=avg_cov)

        def score_fn(x_0):
            nelbo, terms = nelbo_dtdpm(diffusion, x_0, rev_var_type, trajectory=trajectory, sample_steps=sample_steps, ms_eps=ms_eps, nll_terms=nll_terms)
            return tuple([nelbo, *terms])
        outputs = score_on_dataset(dataset, score_fn, batch_size)
        outputs_bpd = [a / (self.dataset.data_dim * math.log(2.)) for a in outputs]
        nelbo_bpd = outputs_bpd[0]
        terms_bpd = outputs_bpd[1:]
        self.interact.report_scalar(nelbo_bpd, it, 'bpd')
        self.interact.report_scalar(sum(terms_bpd[1:]), it, 'continuous_part')
        self.interact.report_scalar(terms_bpd[0], it, 'discrete_part')
        torch.save({"nelbo_bpd": nelbo_bpd, "terms_bpd": terms_bpd}, fname)

    def save_nll_terms(self, fname, batch_size, schedule, rev_var_type, clip_x0=True, avg_cov=False,
                       n_samples=None, partition="test", it=None):
        dataset = self.dataset.get_data(partition, labelled=False)
        n_samples = n_samples or len(dataset)
        idxes = random.sample(list(range(len(dataset))), n_samples)
        dataset = Subset(dataset, idxes)

        logging.info("save_nll_terms with {} {} samples".format(n_samples, partition))

        diffusion = DDPM(self.wrapper, schedule, clip_x0=clip_x0, avg_cov=avg_cov)

        res = get_nelbo_terms(diffusion, dataset, batch_size, rev_var_type)
        torch.save(res, fname)

        N = schedule.N
        terms = [*[res['F'][n, n + 1] for n in range(0, N)], res['last_term']]
        terms_bpd = [a / (self.dataset.data_dim * math.log(2.)) for a in terms]
        nelbo_bpd = sum(terms_bpd)
        self.interact.report_scalar(nelbo_bpd, it, 'bpd')
        self.interact.report_scalar(sum(terms_bpd[1:]), it, 'continuous_part')
        self.interact.report_scalar(terms_bpd[0], it, 'discrete_part')

    def save_ms_eps(self, fname, batch_size, schedule, n_samples=None, partition="train_val", it=None):
        dataset = self.dataset.get_data(partition, labelled=False)
        n_samples = n_samples or len(dataset)
        idxes = random.sample(list(range(len(dataset))), n_samples)
        dataset = Subset(dataset, idxes)

        logging.info("save_ms_eps with {} samples".format(n_samples))

        diffusion = DTDPM(self.wrapper, schedule, clip_x0=True)
        N = diffusion.N

        ms_eps = np.zeros(N + 1, dtype=np.float32)
        vars_ = np.zeros(N + 1, dtype=np.float32)
        ests = []
        for n in range(1, N + 1):
            @ torch.no_grad()
            def score_fn(x0):
                eps = torch.randn_like(x0)
                xn = diffusion.cum_alphas[n] ** 0.5 * x0 + diffusion.cum_betas[n] ** 0.5 * eps
                x0_pred, eps_pred = diffusion.predict_x0_eps(xn, n)
                return func.mos(eps_pred)
            est = cat_score_on_dataset(dataset, score_fn, batch_size)
            ms_eps[n] = est.mean()
            vars_[n] = est.var()
            ests.append(est.cpu().numpy())
            logging.info("[n: {}] [ms_eps[{}]: {}] [vars[{}]: {}]".format(n, n, ms_eps[n], n, vars_[n]))

        path, name = os.path.split(fname)
        torch.save(ms_eps, fname)
        torch.save(vars_, os.path.join(path, "vars_%s" % name))
        torch.save(ests, os.path.join(path, "ests_%s" % name))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.plot(list(range(1, N + 1)), ms_eps[1:])
        plt.savefig("{}.png".format(fname))
        plt.close()
