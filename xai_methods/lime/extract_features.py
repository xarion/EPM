from captum._utils.models import SkLearnLinearRegression
from captum.attr._core.lime import get_exp_kernel_similarity_function, Lime
from torch.utils.data import DataLoader

from config import IMAGE_CLASS
from dataset import get_validation_dataset
from models import create_model, EncodingSavingHook
def main():
    xai_method_name = "Lime"
    ds = get_validation_dataset()
    dl = DataLoader(ds, batch_size=1)

    encoding_saving_hook = EncodingSavingHook(xai_method_name)
    model, features = create_model()
    features.register_forward_hook(encoding_saving_hook.hook)
    exp_eucl_distance = get_exp_kernel_similarity_function('euclidean', kernel_width=1000)

    for i, (images, labels) in enumerate(iter(dl)):
        lr_lime = Lime(
            model,
            interpretable_model=SkLearnLinearRegression(),  # build-in wrapped sklearn Linear Regression
            similarity_func=exp_eucl_distance
        )
        attributions = lr_lime.attribute(images, target=IMAGE_CLASS, n_samples=100, show_progress=True)

    encoding_saving_hook.save_encodings()
if __name__ == "__main__":
    main()
