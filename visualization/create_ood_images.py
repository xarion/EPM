import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from visualization.viz_utils import save_figure
from PIL import Image


image_size = (224, 224, 3)

def create_white_noise_dataset(n=3):

    seeds = [i for i in range(n)]

    for s in seeds:
        rs = np.random.RandomState(seed=s)
        image = rs.uniform(low=0, high=255, size=image_size).astype(np.uint8)
        image = Image.fromarray(image, mode="RGB")
        save_figure(image, "wn_%d.jpg" % s, "ood")


def create_anime_dataset():
    # https://www.gwern.net/Danbooru2020#kaggle
    pass

