from features import features
from models import create_feature_extractor_model
from PIL import Image
from glob import glob
import numpy as np

files = glob("lime*.jpg")
images = list()

for file in sorted(files):
    image = Image.open(file)
    image_data = np.asarray(image)
    image_data = np.expand_dims(image_data, 0)
    images.append(image_data)

image_array = np.concatenate(images)
f_1 = features(image_array[:100])
f_2 = features(image_array[100:])

np.save("lime_only_ball_features.npy", f_1)
np.save("lime_man_hitting_ball_features.npy", f_2)