import numpy as np
import torch
from anchor import anchor_image
from torch.utils.data import DataLoader

from dataset import get_validation_dataset
from models import EncodingSavingHook, create_model


def main():
    xai_method_name = "AnchorLime"
    ds = get_validation_dataset()
    dl = DataLoader(ds, batch_size=1)

    encoding_saving_hook = EncodingSavingHook(xai_method_name)
    model, features = create_model()
    features.register_forward_hook(encoding_saving_hook.hook)

    def __predict(_npy_image):
        _npy_image = np.transpose(_npy_image, (0, 3, 1, 2))
        torch_image = torch.from_numpy(_npy_image).float()
        return model(torch_image).detach().cpu().numpy()

    explainer = anchor_image.AnchorImage()

    for i, (images, labels) in enumerate(iter(dl)):
        npy_image = images.detach().numpy()
        npy_image = np.squeeze(npy_image)
        npy_image = np.transpose(npy_image, (1, 2, 0)).astype(float)
        explainer.explain_instance(npy_image, __predict)

    encoding_saving_hook.save_encodings()


if __name__ == "__main__":
    main()
