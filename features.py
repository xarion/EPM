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

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files_list = np.array([f for f in glob(root_dir + "/*.jpg")])
        super().__init__()
        self.transform = T.Compose(
            [T.Resize(256), T.CenterCrop(224), T.Resize(64), T.ToTensor(),
             Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)])

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.files_list[idx])
        image = Image.open(img_name).convert('RGB')
        torch_image_data = self.transform(image)
        torch_image_class = torch.from_numpy(np.array([IMAGE_CLASS]))
        torch_image_class = torch_image_class.float()
        return torch_image_data, torch_image_class
