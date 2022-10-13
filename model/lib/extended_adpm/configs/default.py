import ml_collections
import interface.datasets as datasets
import torch.optim as optim
import os
import datetime
import core.criterions as criterions
import interface.evaluators as evaluators
import core.diffusion.wrapper as wrapper
from torch.optim.lr_scheduler import LambdaLR


################################################################################
# Datasets
################################################################################

def get_cifar10_config(**hparams):
    config = ml_collections.ConfigDict()
    config.use_val = hparams.get('use_val', False)
    config.cls = datasets.CIFAR10
    config.kwargs = kwargs = ml_collections.ConfigDict()
    kwargs.data_path = hparams.get('data_path', 'workspace/datasets/cifar10/')
    kwargs.random_flip = hparams.get('random_flip', False)
    return config


def get_celeba64_config(**hparams):
    config = ml_collections.ConfigDict()
    config.use_val = hparams.get('use_val', False)
    config.cls = datasets.CelebA
    config.kwargs = kwargs = ml_collections.ConfigDict()
    kwargs.data_path = hparams.get('data_path', 'workspace/datasets/celeba/')
    kwargs.width = 64
    return config


def get_imagenet64_config(**hparams):
    config = ml_collections.ConfigDict()
    config.use_val = hparams.get('use_val', False)
    config.cls = datasets.Imagenet64
    config.kwargs = kwargs = ml_collections.ConfigDict()
    kwargs.data_path = hparams.get('data_path', 'workspace/datasets/imagenet64/')
    return config


def get_lsun_bedroom_config(**hparams):
    config = ml_collections.ConfigDict()
    config.use_val = hparams.get('use_val', False)
    config.cls = datasets.LSUNBedroom
    config.kwargs = kwargs = ml_collections.ConfigDict()
    kwargs.data_path = hparams.get('data_path', 'workspace/datasets/lsun_bedroom/')
    return config


################################################################################
# Optimizers
################################################################################

def get_optimizers_config(**hparams):
    config = ml_collections.ConfigDict()
    config.all = ml_collections.ConfigDict()
    config.all.cls = optim.Adam
    config.all.kwargs = ml_collections.ConfigDict()
    config.all.kwargs.lr = hparams.get('lr', 0.0001)
    config.all.kwargs.weight_decay = hparams.get('weight_decay', 0.)
    return config


################################################################################
# LRSchedulers
################################################################################


def customized_lr_scheduler(optimizer, warmup_its):
    def fn(it):
        if warmup_its is None:
            return 1
        else:
            return min(it / warmup_its, 1)
    return LambdaLR(optimizer, fn)


def get_lr_schedulers_config(**hparams):
    config = ml_collections.ConfigDict()
    config.all = ml_collections.ConfigDict()
    config.all.cls = customized_lr_scheduler
    config.all.kwargs = ml_collections.ConfigDict()
    config.all.kwargs.warmup_its = hparams.get('warmup_its', None)
    return config


################################################################################
# Train
################################################################################

def get_train_config(**hparams):
    config = ml_collections.ConfigDict()
    config.seed = hparams.get('seed', 1234)
    config.deterministic = hparams.get('deterministic', False)
    config.workspace = hparams['workspace']
    config.ckpt_root = os.path.join(config.workspace, 'train/ckpts/')
    config.backup_root = os.path.join(config.workspace, 'train/reproducibility/')
    config.date = hparams.get('date', datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))

    config.training = training = ml_collections.ConfigDict()
    training.n_ckpts = hparams.get('n_ckpts', 50)
    training.n_its = hparams.get('n_its', 500000)
    training.batch_size = hparams.get('batch_size', 128)

    config.ema = ema = ml_collections.ConfigDict()
    ema.rate = hparams.get('ema_rate', 0.9999)

    config.optimizers = get_optimizers_config(**hparams)
    config.lr_schedulers = get_lr_schedulers_config(**hparams)

    config.interact = ml_collections.ConfigDict()
    config.interact.fname_log = os.path.join(config.workspace, f'train/logs/{config.date}.log')
    config.interact.summary_root = os.path.join(config.workspace, 'train/summary/')
    config.interact.period = hparams.get('period', 10)
    return config


################################################################################
# Evaluate
################################################################################

def get_grid_sample_config(**hparams):
    config = ml_collections.ConfigDict()
    config.fn = 'grid_sample'
    config.period = 5000
    config.kwargs = kwargs = ml_collections.ConfigDict()
    kwargs.schedule = hparams['schedule']
    kwargs.rev_var_type = hparams.get('rev_var_type', 'small')
    kwargs.path = os.path.join(hparams['workspace'], 'train/evaluator/grid_sample/')
    kwargs.sample_steps = 50
    return config


def get_sde_grid_sample_config(**hparams):
    config = ml_collections.ConfigDict()
    config.fn = 'grid_sample'
    config.period = 5000
    config.kwargs = kwargs = ml_collections.ConfigDict()
    kwargs.sde = hparams['sde']
    kwargs.path = os.path.join(hparams['workspace'], 'train/evaluator/grid_sample/')
    kwargs.sample_steps = 50
    return config


def get_sample2dir_config(**hparams):
    config = ml_collections.ConfigDict()
    config.fn = 'sample2dir'
    config.kwargs = kwargs = ml_collections.ConfigDict()
    kwargs.schedule = hparams['schedule']
    kwargs.forward_type = hparams.get('forward_type', 'ddpm')
    kwargs.rev_var_type = hparams['rev_var_type']
    kwargs.clip_x0 = True
    kwargs.avg_cov = hparams.get('avg_cov', False)
    kwargs.trajectory = hparams.get('trajectory', 'linear')
    kwargs.sample_steps = hparams['sample_steps']
    kwargs.clip_sigma_idx = hparams.get('clip_sigma_idx', 0)
    kwargs.clip_pixel = hparams.get('clip_pixel', 2)
    kwargs.eta = hparams.get('eta', None)
    kwargs.ms_eps_path = hparams.get('ms_eps_path', None)
    return config


def get_sde_sample2dir_config(**hparams):
    config = ml_collections.ConfigDict()
    config.fn = 'sample2dir'
    config.kwargs = kwargs = ml_collections.ConfigDict()
    kwargs.path = hparams['path']
    kwargs.n_samples = hparams['n_samples']
    kwargs.batch_size = hparams['batch_size']
    kwargs.sde = hparams['sde']
    kwargs.reverse_type = hparams.get('reverse_type', 'sde')
    kwargs.sample_steps = hparams['sample_steps']
    return config


def get_nll_config(**hparams):
    config = ml_collections.ConfigDict()
    config.fn = 'nll'
    config.kwargs = kwargs = ml_collections.ConfigDict()
    kwargs.fname = hparams['fname']
    kwargs.batch_size = hparams['batch_size']
    kwargs.schedule = hparams['schedule']
    kwargs.rev_var_type = hparams['rev_var_type']
    kwargs.clip_x0 = True
    kwargs.avg_cov = hparams.get('avg_cov', False)
    kwargs.trajectory = hparams.get('trajectory', 'linear')
    kwargs.sample_steps = hparams['sample_steps']
    kwargs.n_samples = hparams.get('n_samples', None)
    kwargs.partition = hparams.get('hparams', 'test')
    kwargs.ms_eps_path = hparams.get('ms_eps_path', None)
    kwargs.nll_terms_path = hparams.get('nll_terms_path', None)
    return config


def get_sde_nll_config(**hparams):
    config = ml_collections.ConfigDict()
    config.fn = 'nll'
    config.kwargs = kwargs = ml_collections.ConfigDict()
    kwargs.fname = hparams['fname']
    kwargs.batch_size = hparams['batch_size']
    kwargs.sde = hparams['sde']
    kwargs.reverse_type = hparams['reverse_type']
    kwargs.n_samples = hparams.get('n_samples', None)
    kwargs.partition = hparams.get('hparams', 'test')
    kwargs.t_init = hparams.get('t_init', 1e-5)
    return config


def get_save_nll_terms_config(**hparams):
    config = ml_collections.ConfigDict()
    config.fn = 'save_nll_terms'
    config.kwargs = kwargs = ml_collections.ConfigDict()
    kwargs.fname = hparams['fname']
    kwargs.batch_size = hparams['batch_size']
    kwargs.schedule = hparams['schedule']
    kwargs.rev_var_type = hparams['rev_var_type']
    kwargs.clip_x0 = True
    kwargs.n_samples = hparams.get('n_samples', None)
    kwargs.partition = hparams.get('hparams', 'test')
    return config


def get_save_ms_eps_config(**hparams):
    config = ml_collections.ConfigDict()
    config.fn = 'save_ms_eps'
    config.kwargs = kwargs = ml_collections.ConfigDict()
    kwargs.fname = hparams['fname']
    kwargs.batch_size = hparams['batch_size']
    kwargs.schedule = hparams['schedule']
    kwargs.n_samples = hparams.get('n_samples', None)
    kwargs.partition = hparams.get('partition', 'train_val')
    return config


def get_train_evaluator_config(**hparams):
    config = ml_collections.ConfigDict()
    config.cls = evaluators.DTDPMEvaluator
    config.kwargs = kwargs = ml_collections.ConfigDict()
    kwargs.options = options = ml_collections.ConfigDict()
    options.grid_sample = get_grid_sample_config(**hparams)
    return config


def get_sde_train_evaluator_config(**hparams):
    config = ml_collections.ConfigDict()
    config.cls = evaluators.SDEEvaluator
    config.kwargs = kwargs = ml_collections.ConfigDict()
    kwargs.options = options = ml_collections.ConfigDict()
    options.grid_sample = get_sde_grid_sample_config(**hparams)
    return config


################################################################################
# Wrappers
################################################################################

def get_dt_wrapper_config(**hparams):
    config = ml_collections.ConfigDict()
    config.cls = wrapper.DTWrapper
    config.kwargs = kwargs = ml_collections.ConfigDict()
    kwargs.rescale_timesteps = hparams.get('rescale_timesteps', True)
    kwargs.shift1 = hparams.get('shift1', False)
    kwargs.N = hparams['N']
    kwargs.typ = hparams['typ']
    return config


def get_split_dt_wrapper_config(**hparams):
    config = get_dt_wrapper_config(**hparams)
    config.cls = wrapper.SplitDTWrapper
    config.split_idx = hparams['split_idx']
    return config


def get_ct2dt_wrapper_config(**hparams):
    config = ml_collections.ConfigDict()
    config.cls = wrapper.CT2DTWrapper
    config.kwargs = kwargs = ml_collections.ConfigDict()
    kwargs.N = hparams['N']
    kwargs.typ = hparams['typ']
    return config


def get_ct_wrapper_config(**hparams):
    config = ml_collections.ConfigDict()
    config.cls = wrapper.CTWrapper
    config.kwargs = kwargs = ml_collections.ConfigDict()
    kwargs.typ = hparams['typ']
    return config


################################################################################
# Criterions
################################################################################

def get_dt_dsm_config(**hparams):
    config = ml_collections.ConfigDict()
    config.cls = criterions.DTDSM
    config.kwargs = ml_collections.ConfigDict()
    config.kwargs.schedule = hparams['schedule']
    return config


def get_dt_dsm0_config(**hparams):
    config = ml_collections.ConfigDict()
    config.cls = criterions.DTDSMZero
    config.kwargs = ml_collections.ConfigDict()
    config.kwargs.schedule = hparams['schedule']
    config.kwargs.weighted = hparams.get('weighted', True)
    return config


def get_dt_dsdm_config(**hparams):
    config = ml_collections.ConfigDict()
    config.cls = criterions.DTDSDM
    config.kwargs = ml_collections.ConfigDict()
    config.kwargs.schedule = hparams['schedule']
    config.kwargs.ratio = hparams.get('ratio', 1)
    return config


def get_dt_dsdm_err_config(**hparams):
    config = ml_collections.ConfigDict()
    config.cls = criterions.DTDSDMErr
    config.kwargs = ml_collections.ConfigDict()
    config.kwargs.schedule = hparams['schedule']
    config.kwargs.ratio = hparams.get('ratio', 1)
    return config


def get_ct_dsdm_config(**hparams):
    config = ml_collections.ConfigDict()
    config.cls = criterions.CTDSDM
    config.kwargs = ml_collections.ConfigDict()
    config.kwargs.sde = hparams['sde']
    config.kwargs.ratio = hparams.get('ratio', 1)
    return config


def get_ct_dsdm_err_config(**hparams):
    config = ml_collections.ConfigDict()
    config.cls = criterions.CTDSDMErr
    config.kwargs = ml_collections.ConfigDict()
    config.kwargs.sde = hparams['sde']
    config.kwargs.ratio = hparams.get('ratio', 1)
    return config


def get_lhybrid_config(**hparams):
    config = ml_collections.ConfigDict()
    config.cls = criterions.LHybrid
    config.kwargs = ml_collections.ConfigDict()
    config.kwargs.schedule = hparams['schedule']
    return config


def get_ct_dsm_config(**hparams):
    config = ml_collections.ConfigDict()
    config.cls = criterions.CTDSM
    config.kwargs = ml_collections.ConfigDict()
    config.kwargs.sde = hparams['sde']
    config.kwargs.grad_clip = hparams.get('grad_clip', None)
    return config
