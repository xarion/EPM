import json

import numpy as np
import shap
from lime import lime_image

from dataset import get_validation_dataset
from models import create_adjusted_image_saving_model_for_xai

xai_method_name = "lime"

val_ds = get_validation_dataset(batch_size=20)
np_val_ds = val_ds.__iter__().next().numpy()
to_train = np_val_ds[2:, ...]
to_explain = np_val_ds[:2, ...]

# load the ImageNet class names
url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
fname = shap.datasets.cache(url)
with open(fname) as f:
    class_names = json.load(f)

explainer = lime_image.LimeImageExplainer(random_state=420)

for i in range(0, to_explain.shape[0]):
    model = create_adjusted_image_saving_model_for_xai(f"{xai_method_name}_{i}")
    explainer.explain_instance(to_explain[i].astype(np.double),
                               model.predict,
                               top_labels=5, hide_color=0, num_samples=1000,
                               random_seed=42)
