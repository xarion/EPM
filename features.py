import os
from glob import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import Normalize

from config import IMAGE_CLASS, USE_CUDA, IMAGE_MEAN, IMAGE_STD
from models import create_model, EncodingSavingHook


def save_features(dataset, name):
    encoding_saving_hook = EncodingSavingHook(name)
    model, features = create_model()
    features.register_forward_hook(encoding_saving_hook.hook)

    dataloader = DataLoader(dataset, batch_size=16, pin_memory=True)

    for i, (images, labels) in enumerate(iter(dataloader)):
        if USE_CUDA:
            images = images.cuda()
        model(images)
    encoding_saving_hook.save_encodings()


def save_features_from_folder(xai_method, folder="./"):
    dataset = PerturbedImagesDataset(folder)
    save_features(dataset, xai_method)


class PerturbedImagesDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files_list = np.array([f for f in glob("*.jpg") if ".1." not in f])
        super(PerturbedImagesDataset).__init__(transform=T.Compose(
            [T.Resize(256), T.CenterCrop(224), T.Resize(64), T.ToTensor(), Normalize(mean=IMAGE_MEAN,
                                                                                     std=IMAGE_STD)]))

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.files_list[idx])
        image = Image.open(img_name)
        image_data = np.array(image)
        if len(image_data.shape) == 2:
            image_data = np.expand_dims(image_data, axis=2)
            image_data = np.repeat(image_data, 3, axis=2)
        image_data = image_data.transpose((2, 0, 1)).astype(float)
        torch_image_data = torch.from_numpy(image_data)
        torch_image_data = torch_image_data.float()
        torch_image_class = torch.from_numpy(np.array([IMAGE_CLASS]))
        torch_image_class = torch_image_class.float()
        return torch_image_data, torch_image_class
