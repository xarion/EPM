from captum.attr import KernelShap
from torch.utils.data import DataLoader

from config import USE_CUDA, tennis_ball_class
from dataset import get_validation_dataset
from models import create_model, EncodingSavingHook


def main(model_name, image_class):
    xai_method_name = "KernelShap"
    ds = get_validation_dataset(image_class)
    dl = DataLoader(ds, batch_size=1, pin_memory=True)

    encoding_saving_hook = EncodingSavingHook(model_name, image_class, xai_method_name)
    model, features = create_model(model_name)
    features.register_forward_hook(encoding_saving_hook.hook)

    for i, (images, labels) in enumerate(iter(dl)):
        if USE_CUDA:
            images = images.cuda()
        shap = KernelShap(model)
        attributions = shap.attribute(images, target=image_class, n_samples=100, show_progress=True)

    encoding_saving_hook.save_encodings()


if __name__ == "__main__":
    main("resnet50", tennis_ball_class)
