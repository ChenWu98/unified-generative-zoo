from .utils import is_labelled, UnlabeledDataset
from torch.utils.data import ConcatDataset
import numpy as np


class DatasetFactory(object):
    r""" Output dataset after two transformations to the raw data:
    1. distribution transform (e.g. binarized, adding noise), often irreversible, a part of which is implemented
       in distribution_transform
    2. an affine transform (preprocess), which is bijective
    """

    def __init__(self):
        self.train = None
        self.val = None
        self.test = None

    def allow_labelled(self):
        return is_labelled(self.train)

    def transform_data(self, dataset, labelled):
        assert not (not is_labelled(dataset) and labelled)
        if is_labelled(dataset) and not labelled:
            dataset = UnlabeledDataset(dataset)
        return self.affine_transform(self.distribution_transform(dataset))

    def get_train_data(self, labelled=False):
        return self.transform_data(self.train, labelled=labelled)

    def get_val_data(self, labelled=False):
        return self.transform_data(self.val, labelled=labelled)

    def get_train_val_data(self, labelled=False):
        train_val = ConcatDataset([self.train, self.val]) if self.val is not None else self.train
        return self.transform_data(train_val, labelled=labelled)

    def get_test_data(self, labelled=False):
        return self.transform_data(self.test, labelled=labelled)

    def get_data(self, partition, labelled=False):
        if partition == "train":
            return self.get_train_data(labelled)
        elif partition == "val":
            return self.get_val_data(labelled)
        elif partition == "test":
            return self.get_test_data(labelled)
        elif partition == "train_val":
            return self.get_train_val_data(labelled)
        else:
            raise ValueError

    def distribution_transform(self, dataset):
        return dataset

    def affine_transform(self, dataset):
        return dataset

    def preprocess(self, v):
        r""" The mathematical form of the affine transform
        """
        return v

    def unpreprocess(self, v):
        r""" The mathematical form of the affine transform's inverse
        """
        return v

    @property
    def data_shape(self):
        raise NotImplementedError

    @property
    def data_dim(self):
        return int(np.prod(self.data_shape))

    @property
    def fid_stat(self):
        return None

    @property
    def cov(self):  # covariance of the training data
        if "_cov" not in self.__dict__:
            tr = self.get_train_data(labelled=False)
            tr = np.array([item.flatten().numpy() for item in tr])
            self.__dict__["_cov"] = np.cov(tr.transpose())
        return self.__dict__["_cov"]

    @property
    def trace_cov(self):
        if "_trace_cov" not in self.__dict__:
            self.__dict__["_trace_cov"] = self.cov.trace()
        return self.__dict__["_trace_cov"]

    @property
    def trace_cov_avg(self):
        return self.trace_cov / self.data_dim
