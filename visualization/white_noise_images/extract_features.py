from features import save_features_from_folder


# MODEL_NAMES = ["mnasnet1.0", "densenet121", "resnet50"]


save_features_from_folder(model_name="resnet50", image_class="", dataset_name="white_noise",
                          folder="/home/gabi/PycharmProjects/EPM/visualization/white_noise_images/dataset")