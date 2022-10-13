import torchvision.transforms as transforms
from .dataset_factory import DatasetFactory
from .lsun.lsun import LSUN
from .utils import *
import os


class LSUNBedroom(DatasetFactory):
    def __init__(self, data_path):
        super().__init__()
        _transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor()])
        self.train = LSUN(root=data_path, classes=["bedroom_train"], transform=_transform) \
            if os.path.exists(os.path.join(data_path, 'bedroom_train_lmdb')) else None
        self.test = LSUN(root=data_path, classes=["bedroom_val"], transform=_transform) \
            if os.path.exists(os.path.join(data_path, 'bedroom_val_lmdb')) else None

    def affine_transform(self, dataset):
        return StandardizedDataset(dataset, mean=0.5, std=0.5)  # scale to [-1, 1]

    def preprocess(self, v):
        return 2. * (v - 0.5)

    def unpreprocess(self, v):
        v = 0.5 * (v + 1.)
        v.clamp_(0., 1.)
        return v

    @property
    def data_shape(self):
        return 3, 256, 256

    @property
    def fid_stat(self):
        return 'workspace/fid_stats/fid_stats_lsun_bedroom_train_50000_ddim.npz'
