from captum.attr import Occlusion
from torch.utils.data import DataLoader

from config import IMAGE_CLASS, USE_CUDA
from dataset import get_validation_dataset
from models import create_model, EncodingSavingHook


def main():
    xai_method_name = "Occlusion"
    ds = get_validation_dataset()
    dl = DataLoader(ds, batch_size=1, pin_memory=True)

    encoding_saving_hook = EncodingSavingHook(xai_method_name)
    model, features = create_model()
    features.register_forward_hook(encoding_saving_hook.hook)

    for i, (images, labels) in enumerate(iter(dl)):
        if USE_CUDA:
            images = images.cuda()
        occlusion = Occlusion(model)
        attributions = occlusion.attribute(images, target=IMAGE_CLASS,
                                           sliding_window_shapes=(3, 6, 6),
                                           strides=(3, 3, 3),
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


if __name__ == "__main__":
    main()
