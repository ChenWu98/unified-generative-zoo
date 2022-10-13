import core.utils.managers as managers
import torch.optim as optim
from .interact import Interact
from core.evaluate import score_on_dataset
import functools
import torch
import logging


def create_instance(config):
    kwargs = config.get("kwargs", {})
    return config.cls(**kwargs)


################################################################################
# Create models from a profile
################################################################################


def create_models(config):
    models = {}
    for k, val in config.items():
        models[k] = create_instance(val)
        path = val.get('pretrained_path', None)
        if path is not None:
            logging.info(f'load pretrained model from {path} for {k}')
            msg = models[k].load_state_dict(torch.load(path), strict=False)
            logging.info(msg)
        else:
            logging.info(f'randomly initialize {k}')
    return managers.ModelsManager(**models)


def create_wrapper(config, models: managers.ModelsManager):
    return config.cls(**models.kwargs, **config.kwargs)


################################################################################
# Create optimizers from a profile
################################################################################

def create_optimizer(config, models: managers.ModelsManager):
    r""" Create an instance of the optimizer described in the profile
    Args:
        config: a config describing the optimizer
            Example: { "cls": optim.Adam,
                       "model_keys": ["lvm", "q"],
                       "kwargs": { "lr": 0.0001 } }
            If 'model_keys' is missing, the corresponding optimizer will include all parameters
        models: an object of ModelsManager
    """
    params = models.parameters(*config.get("model_keys", []))
    return config.cls(params, **config.kwargs)


def create_optimizers(config, models: managers.ModelsManager):
    r""" Create optimizers (an instance of OptimizersManager) described in the profile
    Args:
        config: a config describing optimizers
        models: an object of ModelsManager
    """
    optimizers = {}
    for k, val in config.items():
        optimizers[k] = create_optimizer(val, models)
    return managers.OptimizersManager(**optimizers)


################################################################################
# Create lr_schedulers from a profile
################################################################################

def create_lr_scheduler(config, optimizer: optim.Optimizer):
    r""" Create an instance of the optimizer described in the profile
    Args:
        config: a config describing the optimizer
        optimizer: the optimizer to apply
    """
    return config.cls(optimizer, **config.kwargs)


def create_lr_schedulers(config, optimizers: managers.OptimizersManager):
    r""" Create optimizers (an instance of OptimizersManager) described in the profile
    Args:
        config: a config describing optimizers
        optimizers: an object of OptimizersManager
    """
    lr_schedulers = {}
    for k, val in config.items():
        lr_schedulers[k] = create_lr_scheduler(val, optimizers.get(k))
    return managers.LRSchedulersManager(**lr_schedulers)


################################################################################
# Create criterion from a profile
################################################################################

def create_criterion(config,
                     wrapper,
                     models: managers.ModelsManager,
                     optimizers: managers.OptimizersManager,
                     lr_schedulers: managers.LRSchedulersManager):
    r""" Create an instance of the criterion described in the profile
    Args:
        config: a parsed profile describing the criterion
        wrapper: an object of Wrapper
        models: an object of ModelsManager
        optimizers: an object of OptimizersManager
        lr_schedulers: an object of LRSchedulersManager
    """
    return config.cls(**config.get("kwargs", {}), wrapper=wrapper, models=models,
                      optimizers=optimizers, lr_schedulers=lr_schedulers)


################################################################################
# Create dataset from a profile
################################################################################

def create_dataset(config):
    return create_instance(config)


################################################################################
# Create evaluator from a profile
################################################################################

def create_evaluator(config, wrapper, dataset, interact):
    return config.cls(**config.kwargs, wrapper=wrapper, dataset=dataset, interact=interact)


################################################################################
# Create interact from a profile
################################################################################

def create_interact(config) -> Interact:
    return Interact(**config)


################################################################################
# Create the validation function
################################################################################

def create_val_fn(config, criterion):
    if config.get('disable_val_fn', False):  # no val_fn
        return None
    elif 'val_fn' not in config:  # default val_fn
        return functools.partial(score_on_dataset, score_fn=criterion.default_val_fn,
                                 batch_size=config.training.batch_size)
    else:
        batch_size = config.val_fn.get('batch_size', config.training.batch_size)
        kwargs = config.val_fn.get("kwargs", {})
        apply_to = config.val_fn.get("apply_to", "tensor")
        if apply_to == "tensor":
            def score_fn(v):
                return config.val_fn.fn(models=criterion.models, v=v, **kwargs)
            return functools.partial(score_on_dataset, score_fn=score_fn, batch_size=batch_size)
        elif apply_to == "dataset":
            return functools.partial(config.val_fn.fn, models=criterion.models, batch_size=batch_size)
        else:
            raise ValueError


################################################################################
# Create ema
################################################################################

def create_ema(config, models: managers.ModelsManager):
    if config.get("disable_ema", False):
        return None, None
    else:
        ema_keys = config.ema.get('keys', list(models.keys()))
        ema_rate = config.ema.get('rate', 0.9999)
        ema_models = create_models({key: config.models[key] for key in ema_keys})
        ema_models.ema(models, rate=0)
        return ema_models, ema_rate
