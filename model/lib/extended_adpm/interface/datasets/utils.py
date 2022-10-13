import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


def pad22pow(a):
    assert a % 2 == 0
    bits = a.bit_length()
    ub = 2 ** bits
    pad = (ub - a) // 2
    return pad, ub


def is_labelled(dataset):
    labelled = False
    if isinstance(dataset[0], tuple) and len(dataset[0]) == 2:
        labelled = True
    return labelled


class AddGaussNoise(object):
    def __init__(self, std):
        self.std = std

    def __call__(self, tensor):
        return tensor + self.std * torch.rand_like(tensor).to(tensor.device)


class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )


class UnlabeledDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        x, y = self.dataset[item]
        return x


class StandardizedDataset(Dataset):
    def __init__(self, dataset, mean, std):
        self.dataset = dataset
        self.mean = mean
        self.std = std
        self.std_inv = 1. / std
        self.labelled = is_labelled(dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        if self.labelled:
            x, y = self.dataset[item]
            return self.std_inv * (x - self.mean), y
        else:
            x = self.dataset[item]
            return self.std_inv * (x - self.mean)


class QuickDataset(Dataset):
    def __init__(self, array):
        self.array = array

    def __len__(self):
        return len(self.array)

    def __getitem__(self, item):
        return self.array[item]


class FlattenedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.labelled = is_labelled(dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        if self.labelled:
            x, y = self.dataset[item]
            return x.view(-1), y
        else:
            x = self.dataset[item]
            return x.view(-1)
