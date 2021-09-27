# import tensorflow as tf
# tf.compat.v1.disable_v2_behavior()
import glob
import json

import numpy as np
import shap
import os
from dataset import get_validation_dataset
from models import create_adjusted_image_saving_model_for_xai
xai_method_name = "shap"
model = create_adjusted_image_saving_model_for_xai(xai_method_name)
val_ds = get_validation_dataset(batch_size=20)
np_val_ds = val_ds.__iter__().next().numpy()
to_train = np_val_ds[2:, ...]
to_explain = np_val_ds[:2, ...]

# load the ImageNet class names
url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
fname = shap.datasets.cache(url)
with open(fname) as f:
    class_names = json.load(f)


shap.explainers._deep.deep_tf.op_handlers["FusedBatchNormV3"] = shap.explainers._deep.deep_tf.linearity_1d(0)
shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough

e = shap.DeepExplainer(model, to_train)

file_list = glob.glob(f"{xai_method_name}*.jpg", recursive=False)
for file_path in file_list:
    try:
        os.remove(file_path)
    except OSError:
        print("Error while deleting file")

shap_values, indexes = e.shap_values(to_explain)

# get the names for the classes
index_names = np.vectorize(lambda x: class_names[str(x)][1])(indexes)

# plot the explanations
shap.image_plot(shap_values, to_explain, index_names)
