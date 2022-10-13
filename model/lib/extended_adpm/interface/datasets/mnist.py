from torch.utils.data import Subset
from torchvision import datasets
import torchvision.transforms as transforms
from .dataset_factory import DatasetFactory
from .utils import *


class Mnist(DatasetFactory):
    r""" Mnist dataset

    Information of the raw dataset:
         train: 50,000
         val:   10,000
         test:  10,000
         shape: 1 * 28 * 28
         train mean: 0.1309
         train biased std: 0.3085
    """

    def __init__(self, data_path, width=28, binarized=False, gauss_noise=False, noise_std=0.01,
                 padding=False, normalize=None, flatten=False):
        super().__init__()
        self.binarized = binarized
        self.gauss_noise = gauss_noise
        self.noise_std = noise_std
        self.width = width
        self.padding = padding
        self.normalize = normalize
        self.flattened = flatten
        if self.padding:
            self.pad, self.ub = pad22pow(self.width)
            assert self.pad != 0

        if self.binarized:
            assert not self.gauss_noise
            assert self.normalize is None
        assert self.normalize == "standardize" or self.normalize == "subtract_mean" or self.normalize is None

        self.data_path = data_path
        _transform = [transforms.Resize(self.width), transforms.ToTensor()]
        if self.binarized:
            _transform.append(Binarize())
        if self.gauss_noise:
            _transform.append(AddGaussNoise(self.noise_std))
        im_transform = transforms.Compose(_transform)
        self.train_val = datasets.MNIST(self.data_path, train=True, transform=im_transform, download=True)
        self.train = Subset(self.train_val, list(range(50000)))
        self.val = Subset(self.train_val, list(range(50000, 60000)))
        self.test = datasets.MNIST(self.data_path, train=False, transform=im_transform, download=True)

        self.train_mean = 0.1309
        self.train_std = 0.3085

    def affine_transform(self, dataset):
        if self.padding:
            dataset = PaddedDataset(dataset, pad=self.pad)
        if self.normalize == "standardize":
            dataset = StandardizedDataset(dataset, mean=self.train_mean, std=self.train_std)
        elif self.normalize == "subtract_mean":
            dataset = TranslatedDataset(dataset, delta=-self.train_mean)
        if self.flattened:
            dataset = FlattenedDataset(dataset)
        return dataset

    def preprocess(self, v):
        if self.padding:
            v = F.pad(v, [self.pad] * 4)
        if self.normalize == "standardize":
            v = (v - self.train_mean) / self.train_std
        elif self.normalize == "subtract_mean":
            v = v - self.train_mean
        if self.flattened:
            v = v.flatten(1)
        return v

    def unpreprocess(self, v):
        if self.padding:
            v = v.view(len(v), 1, self.ub, self.ub)
        else:
            v = v.view(len(v), 1, self.width, self.width)
        if self.normalize == "standardize":
            v *= self.train_std
            v += self.train_mean
        if self.normalize == "subtract_mean":
            v += self.train_mean
        if self.padding:
            v = v[..., self.pad:-self.pad, self.pad:-self.pad]
        v.clamp_(0., 1.)
        return v
