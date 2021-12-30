import numpy as np
import torch
from anchor import anchor_image
from torch.utils.data import DataLoader

from config import USE_CUDA, tennis_ball_class
from dataset import get_validation_dataset
from models import create_image_saving_model


def main(model_name, image_class):
    xai_method_name = "AnchorLime"
    ds = get_validation_dataset(image_class)
    dl = DataLoader(ds, batch_size=1, pin_memory=True)

    model = create_image_saving_model(model_name, xai_method_name)

    def __predict(_npy_image):
        _npy_image = np.transpose(_npy_image, (0, 3, 1, 2))
        torch_image = torch.from_numpy(_npy_image).float()
        if USE_CUDA:
            torch_image = torch_image.cuda()
        return model(torch_image).detach().cpu().numpy()

    for i, (images, labels) in enumerate(iter(dl)):
        explainer = anchor_image.AnchorImage()
        npy_image = images.detach().numpy()
        npy_image = np.squeeze(npy_image)
        npy_image = np.transpose(npy_image, (1, 2, 0)).astype(float)
        explainer.explain_instance(npy_image, __predict)


if __name__ == "__main__":
    main("resnet50", tennis_ball_class)
