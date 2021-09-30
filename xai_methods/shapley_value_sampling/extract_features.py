from captum.attr import ShapleyValueSampling
from torch.utils.data import DataLoader

from config import IMAGE_CLASS, USE_CUDA
from dataset import get_validation_dataset
from models import create_model, EncodingSavingHook


def main():
    xai_method_name = "ShapleyValueSampling"
    ds = get_validation_dataset()
    dl = DataLoader(ds, batch_size=50, num_workers=16)

    encoding_saving_hook = EncodingSavingHook(xai_method_name)
    model, features = create_model()
    features.register_forward_hook(encoding_saving_hook.hook)

    for i, (images, labels) in enumerate(iter(dl)):
        if USE_CUDA:
            images = images.cuda()
        shapley_value_sampling = ShapleyValueSampling(model)
        shapley_value_sampling.attribute(images, target=IMAGE_CLASS, n_samples=1, show_progress=True)

    encoding_saving_hook.save_encodings()


if __name__ == "__main__":
    main()
