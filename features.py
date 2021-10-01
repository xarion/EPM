import os
from glob import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import Normalize

from config import USE_CUDA, IMAGE_MEAN, IMAGE_STD
from dataset import get_standard_image_transform
from models import create_model, EncodingSavingHook


def save_features(model_name, image_class, dataset, dataset_name):
    encoding_saving_hook = EncodingSavingHook(dataset_name, model_name, image_class)
    model, features = create_model(model_name)
    features.register_forward_hook(encoding_saving_hook.hook)

    dataloader = DataLoader(dataset, batch_size=50, pin_memory=True)

    for i, (images, labels) in enumerate(iter(dataloader)):
        if USE_CUDA:
            images = images.cuda()
        model(images)
    encoding_saving_hook.save_encodings()


def save_features_from_folder(model_name, image_class, dataset_name, folder="./"):
    dataset = PerturbedImagesDataset(folder)
    save_features(model_name, image_class, dataset, dataset_name)


class PerturbedImagesDataset(Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files_list = np.array([f for f in glob(root_dir + "/*.jpg")])
        super().__init__()
        self.transform = get_standard_image_transform()

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.files_list[idx])
        image = Image.open(img_name).convert('RGB')
        torch_image_data = self.transform(image)
        return torch_image_data, None
