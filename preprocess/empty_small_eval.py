# Created by Chen Henry Wu
import torch
from datasets import DatasetDict
from torch.utils.data import Dataset


class Preprocessor(object):

    def __init__(self, args, meta_args):
        self.args = args
        self.meta_args = meta_args

    def preprocess(self, raw_datasets: DatasetDict, cache_root: str):
        assert len(raw_datasets) == 3  # Not always.
        train_dataset = TrainDataset(self.args, self.meta_args, raw_datasets['train'], cache_root)
        dev_dataset = DevDataset(self.args, self.meta_args, raw_datasets['validation'], cache_root)
        test_dataset = TestDataset(self.args, self.meta_args, raw_datasets['test'], cache_root)

        return {
            'train': train_dataset,
            'dev': dev_dataset,
            'test': test_dataset,
        }


class TrainDataset(Dataset):

    def __init__(self, args, meta_args, raw_datasets, cache_root):
        n_sample = 1024

        self.data = [
            {
                "sample_id": torch.LongTensor([idx]).squeeze(0),
                "model_kwargs": ["sample_id", ]
            }
            for idx in range(n_sample)
        ]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class DevDataset(Dataset):

    def __init__(self, args, meta_args, raw_datasets, cache_root):
        n_sample = 32

        self.data = [
            {
                "sample_id": torch.LongTensor([idx]).squeeze(0),
                "model_kwargs": ["sample_id", ]
            }
            for idx in range(n_sample)
        ]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class TestDataset(Dataset):

    def __init__(self, args, meta_args, raw_datasets, cache_root):
        n_sample = 64

        self.data = [
            {
                "sample_id": torch.LongTensor([idx]).squeeze(0),
                "model_kwargs": ["sample_id", ]
            }
            for idx in range(n_sample)
        ]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
