import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms as T

from config import IMAGE_CLASS


def get_validation_dataset():
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    ds = datasets.ImageNet("~/tensorflow_datasets/downloads/manual/", split="val", transform=transform)

    classes = torch.tensor([IMAGE_CLASS])
    indices = (torch.tensor(ds.targets)[..., None] == classes).any(-1).nonzero(as_tuple=True)[0]
    ds = Subset(ds, indices)

    return ds


def get_train_dataset():
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    ds = datasets.ImageNet("~/tensorflow_datasets/downloads/manual/", split="train", transform=transform)

    classes = torch.tensor([IMAGE_CLASS])
    indices = (torch.tensor(ds.targets)[..., None] == classes).any(-1).nonzero(as_tuple=True)[0]
    ds = Subset(ds, indices)
    ds = Subset(ds, torch.tensor(range(0, 250)))

    return ds
