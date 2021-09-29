import os
from glob import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from config import IMAGE_CLASS
from models import create_model


def features(dataset):
    feature_outputs = None

    def _get_feature_outputs():
        def _hook(_module, _input, _output):
            nonlocal feature_outputs
            feature_outputs = _output.detach()

        return _hook

    model, feature_layer = create_model()
    feature_layer.register_forward_hook(_get_feature_outputs())
    features = []
    dl = DataLoader(dataset, batch_size=32, num_workers=16)

    for images, labels in iter(dl):
        model(images)
        features.append(feature_outputs.detach().numpy())

    return np.squeeze(np.concatenate(features, axis=0))


def features_from_modified_inputs(xai_method):
    dataset = PerturbedImagesDataset('./')

    extracted_features = features(dataset)

    np.save(f"{xai_method}.npy", extracted_features)
    return extracted_features


class PerturbedImagesDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files_list = np.array([f for f in glob("*.jpg") if ".1." not in f])

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
