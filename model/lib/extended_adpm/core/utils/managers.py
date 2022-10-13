r""" Sometimes, we need manage multiple pytorch objects in a script, e.g., multiple models, multiple optimizers
    Manager provide a interface to manage them together
"""
import torch.nn as nn
import torch.optim as optim
from .ema import ema
import logging
from core.utils.compatible import Wrapper


class Manager(object):
    def __init__(self, **kwargs):
        r""" Manage a dict of objects
        """
        self.kwargs = kwargs

    def __contains__(self, key):
        return key in self.kwargs

    def keys(self):
        return self.kwargs.keys()

    def items(self):
        return self.kwargs.items()

    def get(self, key: str):
        assert isinstance(key, str)
        return self.kwargs[key]

    def values(self):
        return self.kwargs.values()

    def load_state(self, key: str, state):
        r"""
        Args:
            key: the key of the object
            state: the state of the object
        """
        assert isinstance(key, str)
        logging.info("load {}".format(key))
        self.kwargs[key].load_state_dict(state)

    def get_state(self, key: str):
        assert isinstance(key, str)
        return self.kwargs[key].state_dict()

    def load_states(self, states: dict, *keys):
        r"""
        Args:
            states: a dict of states of objects
            keys: the keys of objects
                If empty, load states for all objects
        """
        assert all(map(lambda x: isinstance(x, str), keys))
        if len(keys) == 0:
            keys = list(self.kwargs.keys())
        for key in keys:
            if key in states:
                self.load_state(key, states[key])
            else:
                logging.info(f"skip {key}")

    def get_states(self, *keys):
        r"""
        Args:
            keys: the keys of objects
                If empty, return the states of all objects
        """
        assert all(map(lambda x: isinstance(x, str), keys))
        if len(keys) == 0:
            keys = list(self.kwargs.keys())
        states = {}
        for key in keys:
            states[key] = self.get_state(key)
        return states


class ModelsManager(Manager):
    def __init__(self, **kwargs):
        r""" Manage a dict of models (nn.Modules)
        """
        for key, model in kwargs.items():
            assert isinstance(model, nn.Module), model
        super().__init__(**kwargs)

    def parameters(self, *keys):
        r""" Return the parameters of models corresponding to keys
            If keys are empty, return the parameters of all models

        Args:
            keys: the keys of models
                If empty, return the parameters of all models
        """
        assert all(map(lambda x: isinstance(x, str), keys))
        if len(keys) == 0:
            keys = list(self.kwargs.keys())
        params = []
        for key in keys:
            params += self.kwargs[key].parameters()
        return params

    def toggle_grad(self, *keys):
        r""" Open the gradient of models corresponding to keys
            Others' gradients will be closed

        Args:
            keys: the keys of models
        """
        assert all(map(lambda x: isinstance(x, str), keys))
        for key, model in self.kwargs.items():
            model.requires_grad_(key in keys)

    def to(self, device):
        for key, model in self.kwargs.items():
            model.to(device)

    def train(self):
        for key, model in self.kwargs.items():
            model.train()

    def eval(self):
        for key, model in self.kwargs.items():
            model.eval()

    def ema(self, src, *keys, rate):
        r""" Exponential moving average
            theta <- beta * theta + (1 - beta) * theta_src

        Args:
            src: the source model
            keys: the keys of models
                If empty, update parameters of all models
            rate: theta <- rate * theta + (1 - rate) * theta_src
        """
        assert isinstance(src, ModelsManager)
        assert all(map(lambda x: isinstance(x, str), keys))
        if len(keys) == 0:
            keys = list(self.kwargs.keys())
        for key in keys:
            ema(self.kwargs[key], src.kwargs[key], rate)


class OptimizersManager(Manager):
    def __init__(self, **kwargs):
        r""" Manage a dict of optimizers (optim.Optimizer)
        """
        assert all(map(lambda obj: isinstance(obj, optim.Optimizer), kwargs.values()))
        super(OptimizersManager, self).__init__(**kwargs)


class LRSchedulersManager(Manager):
    def __init__(self, **kwargs):
        r""" Manage a dict of lr_schedulers (optim.lr_scheduler._LRScheduler)
        """
        assert all(map(lambda obj: isinstance(obj, optim.lr_scheduler._LRScheduler), kwargs.values()))
        super(LRSchedulersManager, self).__init__(**kwargs)
