import os
import sys
sys.path.append(os.path.abspath('model/lib/extended_adpm'))

import math
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import ml_collections

from misc import str2bool, parse_sde, parse_schedule
from core.diffusion.dtdpm import DDPM, DDIM, DTDPM
from interface.utils import ckpt, config_utils
import core.func as func
from core.diffusion.trajectory import _choice_steps
# from core.diffusion.utils import report_statistics
from ..model_utils import requires_grad


def prepare_extended_adpm(source_model_type, method, sample_steps, forward_type, eta):
    print('First of all, when the code changes, make sure that no part in the model is under no_grad!')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--phase', type=str, required=True)
    parser.add_argument('--description', type=str)
    parser.add_argument('--ckpt', type=str, default='best')

    hparams_types = dict(
        pretrained_path=str,  # path to evaluated model
        method=str,
        sample_steps=int,
        n_samples=int,
        batch_size=int,  # the total batch (over all devices)
        seed=int,
        # hyperparameters of architecture
        mode=str,
        # hyperparameters of DPMs with discrete timesteps
        schedule=str,
        rev_var_type=str,
        forward_type=str,
        eta=float,
        trajectory=str,
        clip_sigma_idx=int,
        clip_pixel=int,
        avg_cov=str2bool,
        ms_eps_path=str,
        # hyperparameters of DPMs with continuous timesteps (SDE)
        sde=str,
        reverse_type=str,
        t_init=float,
    )
    for hparam, typ in hparams_types.items():
        parser.add_argument(f'--{hparam}', type=typ)

    if source_model_type == 'celeba64':
        if method == 'pred_eps_eps2_pretrained':
            pretrained_path = 'ckpts/extended_adpm/celeba64_ema_eps_eps2_pretrained_340000.ckpt.pth'
        elif method == 'pred_eps_epsc_pretrained':
            pretrained_path = 'ckpts/extended_adpm/celeba64_ema_eps_epsc_pretrained_190000.ckpt.pth'
        else:
            raise NotImplementedError()
        if forward_type == 'ddpm':
            args = parser.parse_args(
                [
                    '--pretrained_path', pretrained_path,
                    '--dataset', 'celeba64',
                    '--phase', 'sample4test',
                    '--sample_steps', str(sample_steps),
                    '--method', method,
                    '--rev_var_type', 'optimal',
                    '--clip_sigma_idx', str(1),
                    '--clip_pixel', str(2),
                ]
            )
        elif forward_type == 'ddim':
            args = parser.parse_args(
                [
                    '--pretrained_path', pretrained_path,
                    '--dataset', 'celeba64',
                    '--phase', 'sample4test',
                    '--sample_steps', str(sample_steps),
                    '--method', method,
                    '--rev_var_type', 'optimal',
                    '--clip_sigma_idx', str(1),
                    '--clip_pixel', str(1),
                    '--forward_type', 'ddim',
                    '--eta', str(eta),
                ]
            )
    else:
        raise ValueError()

    args.hparams = {key: getattr(args, key) for key in hparams_types.keys() if getattr(args, key) is not None}
    if 'schedule' in args.hparams:
        args.hparams['schedule'] = parse_schedule(args.hparams['schedule'])
    if 'sde' in args.hparams:
        args.hparams['sde'] = parse_sde(args.hparams['sde'])

    if args.dataset == 'cifar10':
        from configs.cifar10 import get_evaluate_config
    elif args.dataset == 'celeba64':
        from configs.celeba64 import get_evaluate_config
    elif args.dataset == 'imagenet64':
        from configs.imagenet64 import get_evaluate_config
    elif args.dataset == 'lsun_bedroom':
        from configs.lsun import get_evaluate_config
    else:
        raise NotImplementedError

    if args.phase == 'sample4test':
        args.hparams.setdefault('n_samples', 50000)  # 5w samples for FID by default
        hparams = {**args.hparams, 'pretrained_path': args.pretrained_path, **args.hparams}
        config = get_evaluate_config(task='sample2dir', **hparams)
    elif args.phase == "nll4test":
        raise ValueError()
    else:
        raise NotImplementedError

    return config


def sample_dtdpm(dtdpm, x_init, rev_var_type, trajectory='linear', sample_steps=None, clip_sigma_idx=0, clip_pixel=2, ms_eps=None):
    r"""
    Sample from the reverse model p(x0|x1)...p(xN-1|xN)p(xN) proposed in DDPM, DDIM and Analytic-DPM
    """
    assert isinstance(dtdpm, DTDPM)
    N = dtdpm.N
    sample_steps = sample_steps or N
    ns = _choice_steps(N, sample_steps, trajectory, ms_eps=ms_eps, betas=dtdpm.betas)
    timesteps = [0] + ns
    # logging.info("sample_dtdpm with rev_var_type={}, trajectory={}, sample_steps={}, clip_sigma_idx={}, clip_pixel={}"
    #              .format(rev_var_type, trajectory, sample_steps, clip_sigma_idx, clip_pixel))
    return _sample_dtdpm(dtdpm, x_init, rev_var_type, timesteps, clip_sigma_idx, clip_pixel, ms_eps)


def dpm_encoder_extended_adpm(dtdpm, image, rev_var_type, trajectory='linear', sample_steps=None, clip_sigma_idx=0, clip_pixel=2, ms_eps=None, white_box_steps=None, forward_type=None):
    assert isinstance(dtdpm, DTDPM)
    N = dtdpm.N
    sample_steps = sample_steps or N
    ns = _choice_steps(N, sample_steps, trajectory, ms_eps=ms_eps, betas=dtdpm.betas)
    timesteps = [0] + ns
    # logging.info("sample_dtdpm with rev_var_type={}, trajectory={}, sample_steps={}, clip_sigma_idx={}, clip_pixel={}"
    #              .format(rev_var_type, trajectory, sample_steps, clip_sigma_idx, clip_pixel))
    return _dpm_encoder_extended_adpm(dtdpm, image, rev_var_type, timesteps, clip_sigma_idx, clip_pixel, ms_eps, white_box_steps=white_box_steps, forward_type=forward_type)


def _dpm_encoder_extended_adpm(dtdpm, image, rev_var_type, timesteps, clip_sigma_idx=0, clip_pixel=2, ms_eps=None, white_box_steps=None, forward_type=None):
    assert isinstance(dtdpm, DTDPM)
    assert timesteps[0] == 0
    print(timesteps, len(timesteps))

    x0 = image
    T = timesteps[-1]
    xT = sample_xT(x0=x0, T=T, dtdpm=dtdpm)
    z_list = [xT, ]

    xt = xT
    idx = 0
    for s, t in list(zip(timesteps, timesteps[1:]))[::-1]:
        # dtdpm.statistics = {}

        # Sample xs.
        print('s, t:', s, t)
        if s != 0:
            a_s = dtdpm.cum_alphas[s]
            at = dtdpm.cum_alphas[t]

            if forward_type == 'ddpm':
                w0 = np.sqrt(a_s) * (1 - at / a_s) / (1 - at)
                wt = np.sqrt(at / a_s) * (1 - a_s) / (1 - at)
                mean = w0 * x0 + wt * xt

                var = (1 - at / a_s) * (1 - a_s) / (1 - at)

                xs = mean + np.sqrt(var) * torch.randn_like(x0)
            elif forward_type == 'ddim':
                et = (xt - np.sqrt(at) * x0) / np.sqrt(1 - at)  # posterior et given x0 and xt
                c1 = dtdpm.eta * np.sqrt((1 - at / a_s) * (1 - a_s) / (1 - at))  # sigma_t
                c2 = np.sqrt((1 - a_s) - c1 ** 2)  # direction pointing to x_t
                xs = np.sqrt(a_s) * x0 + c2 * et + c1 * torch.randn_like(x0)
            else:
                raise ValueError()
        else:
            xs = x0

        # Compute eps.
        x_mean, sigma2 = dtdpm.predict_xprev_cov_xprev(xt, s, t, rev_var_type, ms_eps)

        if s <= timesteps[clip_sigma_idx]:  # clip_sigma_idx = 0 <=> not clip
            # dtdpm.statistics['sigma2_unclip'] = sigma2.mean().item()
            sigma2_threshold = (clip_pixel * 2. / 255. * (math.pi / 2.) ** 0.5) ** 2
            sigma2 = sigma2.clamp(0., sigma2_threshold)
            # dtdpm.statistics['sigma2_threshold'] = sigma2_threshold

        if idx < white_box_steps - 1:
            assert s != 0
            eps = (xs - x_mean) / (sigma2 ** 0.5)
            print('(eps ** 2).sum().item():', idx, (eps ** 2).sum().item())
            # dtdpm.statistics['sigma2'] = sigma2.mean().item()
            z_list.append(eps)
            # report_statistics(s, t, dtdpm.statistics)
            idx += 1
        else:
            break

        xt = xs

    return z_list


def sample_xT(x0, T, dtdpm):
    aT = dtdpm.cum_alphas[T]
    xT = np.sqrt(aT) * x0 + np.sqrt(1 - aT) * torch.randn_like(x0)
    return xT


def sample_dtdpm_with_eps(dtdpm, x_init, rev_var_type, eps_list, trajectory='linear', sample_steps=None, clip_sigma_idx=0, clip_pixel=2, ms_eps=None):
    r"""
    Sample from the reverse model p(x0|x1)...p(xN-1|xN)p(xN) proposed in DDPM, DDIM and Analytic-DPM
    """
    assert isinstance(dtdpm, DTDPM)
    N = dtdpm.N
    sample_steps = sample_steps or N
    ns = _choice_steps(N, sample_steps, trajectory, ms_eps=ms_eps, betas=dtdpm.betas)
    timesteps = [0] + ns
    # logging.info("sample_dtdpm with rev_var_type={}, trajectory={}, sample_steps={}, clip_sigma_idx={}, clip_pixel={}"
    #              .format(rev_var_type, trajectory, sample_steps, clip_sigma_idx, clip_pixel))
    return _sample_dtdpm_with_eps(dtdpm, x_init, rev_var_type, timesteps, clip_sigma_idx, clip_pixel, ms_eps, eps_list=eps_list)


def _sample_dtdpm_with_eps(dtdpm, x_init, rev_var_type, timesteps, clip_sigma_idx=0, clip_pixel=2, ms_eps=None, eps_list=None):
    assert isinstance(dtdpm, DTDPM)
    assert timesteps[0] == 0
    x = x_init

    idx = 0
    for s, t in list(zip(timesteps, timesteps[1:]))[::-1]:
        # dtdpm.statistics = {}
        x_mean, sigma2 = dtdpm.predict_xprev_cov_xprev(x, s, t, rev_var_type, ms_eps)
        if s != 0:
            if s <= timesteps[clip_sigma_idx]:  # clip_sigma_idx = 0 <=> not clip
                # dtdpm.statistics['sigma2_unclip'] = sigma2.mean().item()
                sigma2_threshold = (clip_pixel * 2. / 255. * (math.pi / 2.) ** 0.5) ** 2
                sigma2 = sigma2.clamp(0., sigma2_threshold)
                # dtdpm.statistics['sigma2_threshold'] = sigma2_threshold

            if idx < eps_list.shape[1]:
                x = x_mean + sigma2 ** 0.5 * eps_list[:, idx]
                idx += 1
            else:
                x = x_mean + sigma2 ** 0.5 * torch.randn_like(x)
            # dtdpm.statistics['sigma2'] = sigma2.mean().item()
        else:
            x = x_mean
        # report_statistics(s, t, dtdpm.statistics)
    assert idx == eps_list.shape[1]
    return x


def _sample_dtdpm(dtdpm, x_init, rev_var_type, timesteps, clip_sigma_idx=0, clip_pixel=2, ms_eps=None):
    assert isinstance(dtdpm, DTDPM)
    assert timesteps[0] == 0
    x = x_init
    for s, t in list(zip(timesteps, timesteps[1:]))[::-1]:
        # dtdpm.statistics = {}
        x_mean, sigma2 = dtdpm.predict_xprev_cov_xprev(x, s, t, rev_var_type, ms_eps)
        if s != 0:
            if s <= timesteps[clip_sigma_idx]:  # clip_sigma_idx = 0 <=> not clip
                # dtdpm.statistics['sigma2_unclip'] = sigma2.mean().item()
                sigma2_threshold = (clip_pixel * 2. / 255. * (math.pi / 2.) ** 0.5) ** 2
                sigma2 = sigma2.clamp(0., sigma2_threshold)
                # dtdpm.statistics['sigma2_threshold'] = sigma2_threshold
            x = x_mean + sigma2 ** 0.5 * torch.randn_like(x)
            # dtdpm.statistics['sigma2'] = sigma2.mean().item()
        else:
            x = x_mean
        # report_statistics(s, t, dtdpm.statistics)
    return x


def _rescale_timesteps(n, N, flag):
    if flag:
        return n * 1000.0 / float(N)
    return n


class DTWrapper(object):  # for compatible in models with discrete timesteps (DT)
    r"""
        The forward process is q(x_0, x_1, ..., x_N), which is indexed from 0 to N
        Some codes use different indexes, such as q(x_-1, x_0, ..., x_N-1)
    """
    def __init__(self, model: nn.Module, typ: str, rescale_timesteps: bool, shift1: bool, N, bipartition=None):
        r"""
        Args:
            shift1: whether to shift the index
        """
        super().__init__()
        self.model = model
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


class ExtendedAnalyticDPMWrapper(torch.nn.Module):

    def __init__(self, source_model_type, method, sample_steps, forward_type, white_box_steps, eta=None):
        super(ExtendedAnalyticDPMWrapper, self).__init__()

        self.method = method
        self.white_box_steps = white_box_steps

        # Set up generator
        config = prepare_extended_adpm(source_model_type, method, sample_steps, forward_type, eta)
        # print("config:", config)

        config = ml_collections.FrozenConfigDict(config)
        models = config_utils.create_models(config.models)  # use pretrained_path to load models
        self.generator = models.kwargs['model']
        self.wrapper = DTWrapper(**models.kwargs, **config.wrapper.kwargs)

        self.sample_kwargs = config.evaluator.kwargs.options.sample2dir.kwargs

        self.resolution = config.models.model.kwargs.resolution
        self.in_channels = config.models.model.kwargs.in_channels
        self.latent_dim = self.resolution ** 2 * self.in_channels * self.white_box_steps

        self.ms_eps = None
        self.ms_eps_path = None

        # Freeze.
        requires_grad(self.generator, False)

        # Post process.
        self.post_process = transforms.Compose(  # To un-normalize from [-1.0, 1.0] (GAN output) to [0, 1]
            [transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])]
        )

    def encode(self, image):
        with torch.no_grad():
            # Normalize.
            image = (image - 0.5) * 2.0

            # Resize.
            assert image.shape[2] == image.shape[3] == self.resolution

            z = self._encode(image, **self.sample_kwargs)
            return z

    def _encode(self, image, schedule, forward_type, rev_var_type, clip_x0=True, avg_cov=False,
                trajectory='linear', sample_steps=None, clip_sigma_idx=0, clip_pixel=2, eta=None,
                ms_eps_path=None, persist=True):
        if forward_type == "ddpm":
            diffusion = DDPM(self.wrapper, schedule, clip_x0=clip_x0, avg_cov=avg_cov)
        elif forward_type == "ddim":
            diffusion = DDIM(self.wrapper, schedule, clip_x0=clip_x0, eta=eta, avg_cov=avg_cov)
        else:
            raise NotImplementedError

        if ms_eps_path is not None and self.ms_eps is not None:
            logging.info("load ms_eps from {}".format(ms_eps_path))
            self.ms_eps = torch.load(ms_eps_path)
            self.ms_eps_path = ms_eps_path
        else:
            assert self.ms_eps_path == ms_eps_path

        z_list = dpm_encoder_extended_adpm(diffusion, image, rev_var_type, trajectory=trajectory, sample_steps=sample_steps,
                                           clip_sigma_idx=clip_sigma_idx, clip_pixel=clip_pixel, ms_eps=self.ms_eps, white_box_steps=self.white_box_steps,
                                           forward_type=forward_type)
        bsz = image.shape[0]
        z = torch.stack(z_list, dim=1).view(bsz, -1)
        assert z.shape[1] == self.latent_dim
        return z

    def generate(self, z, schedule, forward_type, rev_var_type, clip_x0=True, avg_cov=False,
                 trajectory='linear', sample_steps=None, clip_sigma_idx=0, clip_pixel=2, eta=None,
                 ms_eps_path=None, persist=True):
        if ms_eps_path is not None and self.ms_eps is not None:
            logging.info("load ms_eps from {}".format(ms_eps_path))
            self.ms_eps = torch.load(ms_eps_path)
            self.ms_eps_path = ms_eps_path
        else:
            assert self.ms_eps_path == ms_eps_path

        bsz = z.shape[0]
        eps_list = z.view(
            bsz,
            self.white_box_steps,
            self.in_channels,
            self.resolution,
            self.resolution,
        )
        x_init = eps_list[:, 0]
        eps_list = eps_list[:, 1:]

        if forward_type == "ddpm":
            diffusion = DDPM(self.wrapper, schedule, clip_x0=clip_x0, avg_cov=avg_cov)
        elif forward_type == "ddim":
            diffusion = DDIM(self.wrapper, schedule, clip_x0=clip_x0, eta=eta, avg_cov=avg_cov)
        else:
            raise NotImplementedError

        img = sample_dtdpm_with_eps(diffusion, x_init, rev_var_type, eps_list, trajectory=trajectory, sample_steps=sample_steps,
                                    clip_sigma_idx=clip_sigma_idx, clip_pixel=clip_pixel, ms_eps=self.ms_eps)

        return img

    def forward(self, z):
        # Eval mode for the generator.
        self.generator.eval()

        img = self.generate(z, **self.sample_kwargs)

        # Post process.
        img = self.post_process(img)

        return img

    @property
    def device(self):
        return next(self.parameters()).device




