from glob import glob

import numpy as np
from PIL import Image

from models import create_feature_extractor_model


def features(dataset):
    model = create_feature_extractor_model()
    return model.predict(dataset)


def features_from_modified_inputs(xai_method, suffix):
    files = glob(f"{xai_method}{suffix}*.jpg")
    images = list()

    for file in sorted(files):
        image = Image.open(file)
        image_data = np.asarray(image)
        image_data = np.expand_dims(image_data, 0)
        images.append(image_data)

    image_array = np.concatenate(images)
    extracted_features = features(image_array)

    np.save(f"{xai_method}{suffix}.npy", extracted_features)
    return extracted_features
