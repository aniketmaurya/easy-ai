import functools
import operator
import os
import pathlib
from glob import glob
from os import path

import torch
from PIL import Image
from torch import mode, optim
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader, Dataset, dataset
from torchvision import datasets, models, transforms

_image_extensions = ['jpg', 'jpeg', 'png']


def check_path(path):
    return os.path.exists(path)


tfsms = transforms.Compose([
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


def get_image_dataset(root_dir) -> Dataset:
    ds = datasets.ImageFolder(root_dir, transform=tfsms)
    print(f'classes are {ds.classes}')
    return ds


def get_dataloader(ds, batch_size, shuffle=True, num_workers=4):
    return DataLoader(ds,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers)


def get_model(ds: Dataset, model=models.resnet18, pretrained=True):
    model = model(pretrained)
    num_classes = len(ds.classes)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    return model, criterion, optimizer
