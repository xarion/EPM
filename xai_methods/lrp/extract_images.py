from captum.attr import LRP
from torch.utils.data import DataLoader

from config import IMAGE_CLASS
from dataset import get_validation_dataset
from models import create_image_saving_model

xai_method_name = "LRP"
ds = get_validation_dataset()
dl = DataLoader(ds, batch_size=1, num_workers=16)

for i, (images, labels) in enumerate(iter(dl)):
    model = create_image_saving_model(f"{xai_method_name}_{i}")
    lrp = LRP(model)
    lrp.attribute(images, target=IMAGE_CLASS)
