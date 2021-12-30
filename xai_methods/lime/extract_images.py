from captum._utils.models import SkLearnLinearRegression
from captum.attr._core.lime import get_exp_kernel_similarity_function, Lime
from torch.utils.data import DataLoader

from config import USE_CUDA, tennis_ball_class
from dataset import get_validation_dataset
from models import create_model, EncodingSavingHook, create_image_saving_model


def main(model_name, image_class):
    xai_method_name = "Lime"
    ds = get_validation_dataset(image_class)
    dl = DataLoader(ds, batch_size=1, pin_memory=True)

    model = create_image_saving_model(model_name, xai_method_name)
    exp_eucl_distance = get_exp_kernel_similarity_function('euclidean', kernel_width=1000)

    for i, (images, labels) in enumerate(iter(dl)):
        if USE_CUDA:
            images = images.cuda()
        lr_lime = Lime(
            model,
            interpretable_model=SkLearnLinearRegression(),  # build-in wrapped sklearn Linear Regression
            similarity_func=exp_eucl_distance
        )
        attributions = lr_lime.attribute(images, target=image_class, n_samples=100, show_progress=True)



if __name__ == "__main__":
    main("resnet50", tennis_ball_class)
