import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms as T
from torchvision.transforms import Normalize

from config import IMAGE_STD, IMAGE_MEAN


def get_standard_image_transform():
    return T.Compose([T.Resize(256), T.CenterCrop(224),
                      T.ToTensor(), Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)])


def get_validation_dataset(image_class):
    ds = datasets.ImageNet("~/tensorflow_datasets/downloads/manual/", split="val",
                           transform=get_standard_image_transform())

    classes = torch.tensor([image_class])
    indices = (torch.tensor(ds.targets)[..., None] == classes).any(-1).nonzero(as_tuple=True)[0]
    ds = Subset(ds, indices)

    return ds


def get_train_dataset(image_class):
    ds = datasets.ImageNet("~/tensorflow_datasets/downloads/manual/", split="train",
                           transform=get_standard_image_transform())

    classes = torch.tensor([image_class])
    indices = (torch.tensor(ds.targets)[..., None] == classes).any(-1).nonzero(as_tuple=True)[0]
    ds = Subset(ds, indices)
    ds = Subset(ds, torch.tensor(range(0, 250)))

    return ds
