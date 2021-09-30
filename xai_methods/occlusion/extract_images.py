import numpy as np
from captum.attr import Occlusion
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

from config import IMAGE_CLASS
from dataset import get_validation_dataset
from models import create_model, EncodingSavingHook

from captum.attr import visualization as viz

xai_method_name = "Occlusion"
ds = get_validation_dataset()
dl = DataLoader(ds, batch_size=1)

encoding_saving_hook = EncodingSavingHook(xai_method_name)
model, features = create_model()
features.register_forward_hook(encoding_saving_hook.hook)
occlusion = Occlusion(model)

for i, (images, labels) in enumerate(iter(dl)):
    attributions = occlusion.attribute(images, target=IMAGE_CLASS,
                                       sliding_window_shapes=(3, 22, 22),
                                       strides=(0, 11, 11),
                                       show_progress=True)

    # default_cmap = LinearSegmentedColormap.from_list('custom blue',
    #                                                  [(0, '#ffffff'),
    #                                                   (0.25, '#000000'),
    #                                                   (1, '#000000')], N=256)
    # _ = viz.visualize_image_attr(np.transpose(attributions.squeeze().cpu().detach().numpy(), (1, 2, 0)),
    #                              np.transpose(images.squeeze().cpu().detach().numpy(), (1, 2, 0)),
    #                              method='heat_map',
    #                              cmap=default_cmap,
    #                              show_colorbar=True,
    #                              sign='positive',
    #                              outlier_perc=1)


encoding_saving_hook.save_encodings()
