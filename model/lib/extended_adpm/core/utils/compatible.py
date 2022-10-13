__all__ = ["Wrapper"]
import torch.nn as nn


class Wrapper(object):
    def __init__(self, model: [nn.Module, None]):
        self.model_ = model
        self.model = None if model is None else nn.DataParallel(model)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError
