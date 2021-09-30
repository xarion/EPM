from captum.attr import KernelShap
from torch.utils.data import DataLoader

from config import IMAGE_CLASS
from dataset import get_validation_dataset
from models import create_model, EncodingSavingHook


def main():
    xai_method_name = "KernelShap"
    ds = get_validation_dataset()
    dl = DataLoader(ds, batch_size=1)

    encoding_saving_hook = EncodingSavingHook(xai_method_name)
    model, features = create_model()
    features.register_forward_hook(encoding_saving_hook.hook)
    shap = KernelShap(model)

    for i, (images, labels) in enumerate(iter(dl)):
        attributions = shap.attribute(images, target=IMAGE_CLASS, n_samples=100, show_progress=True)

    encoding_saving_hook.save_encodings()


if __name__ == "__main__":
    main()
