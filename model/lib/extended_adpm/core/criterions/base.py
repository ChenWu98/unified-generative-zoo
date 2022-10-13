import core.utils.managers as managers
from core.utils import global_device, diagnose
import torch


class Criterion(object):
    def __init__(self,
                 wrapper,
                 models: managers.ModelsManager,
                 optimizers: managers.OptimizersManager,
                 lr_schedulers: managers.LRSchedulersManager,
                 grad_clip=None,
                 ):
        r""" Criterion does
            1. calculating objectives
            2. calculating gradients
            3. updating parameters

        Args:
            wrapper: an object of Wrapper
            models: an object of ModelsManager
            optimizers: an object of OptimizersManager
            lr_schedulers: an object of LRSchedulersManager
        """
        self.statistics = {}
        self.wrapper = wrapper
        self.models = models
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.grad_clip = grad_clip
        self.device = global_device()

    def objective(self, v, **kwargs):
        raise NotImplementedError

    def update(self, data_loader):
        raise NotImplementedError

    def default_val_fn(self, v):
        r""" Advise a validation function
        """
        return self.objective(v)

    def criterion_name(self):
        return self.__class__.__name__.lower()


class NaiveCriterion(Criterion):
    def update(self, data_loader):
        v = next(data_loader).to(self.device)
        loss = self.objective(v).mean()
        self.statistics[self.criterion_name()] = loss.item()
        self.statistics['lr'] = self.optimizers.get('all').param_groups[0]['lr']
        self.optimizers.get('all').zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.models.parameters(), max_norm=self.grad_clip)
            self.statistics['grad_clip'] = self.grad_clip
        self.optimizers.get('all').step()
        self.lr_schedulers.get('all').step()
