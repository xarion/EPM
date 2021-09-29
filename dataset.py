import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms as T
from torchvision.transforms import Normalize

from config import IMAGE_CLASS, IMAGE_STD, IMAGE_MEAN


def get_validation_dataset():
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), Normalize(mean=IMAGE_MEAN,
                                                                                     std=IMAGE_STD)])
    ds = datasets.ImageNet("/home/erdi/tensorflow_datasets/downloads/manual/", split="val", transform=transform)

    classes = torch.tensor([IMAGE_CLASS])
    indices = (torch.tensor(ds.targets)[..., None] == classes).any(-1).nonzero(as_tuple=True)[0]
    ds = Subset(ds, indices)

    return ds


def get_train_dataset():
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), Normalize(mean=IMAGE_MEAN,
                                                                                     std=IMAGE_STD)])
    ds = datasets.ImageNet("/home/erdi/tensorflow_datasets/downloads/manual/", split="train", transform=transform)

    classes = torch.tensor([IMAGE_CLASS])
    indices = (torch.tensor(ds.targets)[..., None] == classes).any(-1).nonzero(as_tuple=True)[0]
    ds = Subset(ds, indices)
    ds = Subset(ds, torch.tensor(range(0, 250)))

    return ds
