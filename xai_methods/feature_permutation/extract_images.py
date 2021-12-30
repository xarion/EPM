from captum.attr import FeaturePermutation
from torch.utils.data import DataLoader

from config import USE_CUDA, tennis_ball_class
from dataset import get_validation_dataset
from models import create_model, EncodingSavingHook, create_image_saving_model


def main(model_name, image_class):
    xai_method_name = "FeaturePermutation"
    ds = get_validation_dataset(image_class)
    dl = DataLoader(ds, batch_size=50, pin_memory=True)

    model = create_image_saving_model(model_name, xai_method_name)

    for i, (images, labels) in enumerate(iter(dl)):
        if USE_CUDA:
            images = images.cuda()
        feature_permutation = FeaturePermutation(model)
        attributions = feature_permutation.attribute(images, target=image_class, show_progress=True)


if __name__ == "__main__":
    main("resnet50", tennis_ball_class)
