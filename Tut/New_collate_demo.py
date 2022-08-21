import torch
from torch.utils.data import DataLoader
import numpy as np
import glob
from load_dataset_train import LoadDataset
from collections import abc as container_abcs
from batchgenerators.transforms import NumpyToTensor, Compose


"""
1. new_collate is designed for the batchgenerators use
2. A demo shows how the new_collate func works
https://zhuanlan.zhihu.com/p/30385675
"""


def cat_data_demo(catdata):
    return torch.cat(catdata, 0, out=None)


def new_collate_demo(batch_):
    """
    1. Input (batch_) is a list of dataset in batch_size number, elements (batch_[0]) are dict
    2. Aim: Stack all data and add an outer dimension in the batch_size
    """
    print('called new_collate')
    # If data in batch is in dict
    if isinstance(batch_[0], container_abcs.Mapping):
        """
        keys are ['data', 'seg']
        To check generated new dict
        dict_a = {key: cat_data_demo([d[key] for d in batch_]) for key in batch_[0]}
        print(dict_a['seg'].shape)
        """
        return {key: cat_data_demo([d[key] for d in batch_]) for key in batch_[0]}


def load_file_dir():
    train_file_dir = 'dataset/test/images'
    val_file_names = glob.glob('dataset/test_logic/images/*.jpg')
    return train_file_dir, val_file_names


if __name__ == '__main__':
    numpy_to_tensor = NumpyToTensor(['data', 'seg'], cast_to=None)
    tr_transforms = [numpy_to_tensor]
    tr_transforms = Compose(tr_transforms)
    transform = tr_transforms
    train_dir, val_files = load_file_dir()
    liver_dataset = LoadDataset(train_dir, transform)
    train_loader = DataLoader(liver_dataset, batch_size=2, shuffle=True, num_workers=8, pin_memory=True,
                              collate_fn=new_collate_demo)

    for batch in train_loader:
        imgs = batch['data']
        imgs = imgs.to(torch.float32)
        inputs = imgs.cuda(0)
        #print(inputs.shape)