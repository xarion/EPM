from captum.attr import LRP, Occlusion
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from matplotlib import pyplot as plt

from config import IMAGE_CLASS
from dataset import get_validation_dataset
from models import create_image_saving_model, create_model

xai_method_name = "Occlusion"
ds = get_validation_dataset()
dl = DataLoader(ds, batch_size=1, num_workers=16)

for i, (images, labels) in enumerate(iter(dl)):
    # model, features = create_model()
    model = create_image_saving_model(f"{xai_method_name}_{i}")
    occlusion = Occlusion(model)

    attributions = occlusion.attribute(images, target=IMAGE_CLASS, sliding_window_shapes=(3, 20, 20))
    plt.imshow(to_pil_image(attributions[0]))
    plt.show()
    print(attributions)
