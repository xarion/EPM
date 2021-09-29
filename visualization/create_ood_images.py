import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from visualization.viz_utils import save_figure
from PIL import Image
import os


image_size = (224, 224, 3)

def create_white_noise_dataset(n=1):
    # creates uniform white noise

    seeds = [i for i in range(n)]

    for s in seeds:
        rs = np.random.RandomState(seed=s)
        image = rs.uniform(low=0, high=255, size=image_size).astype(np.uint8)
        image = Image.fromarray(image, mode="RGB")
        image.save("white_noise_images/wn_%d.jpg" % s)


def create_anime_dataset(n=1):
    # https://www.gwern.net/Danbooru2020#kaggle
    # make 224x224
    source = "/home/gabi/anime_dataset/danbooru2020/512px/250"
    names = os.listdir(source)[:n]
    if n > 250:
        n = 250

    for name in names:
        p = os.path.join(source, name)
        im = Image.open(p)
        if im.size != (512, 512):
            print("image %s is not 512x512" % name)

        left = (512 - 224) // 2
        top = left
        right = left + 224
        bottom = right
        im = im.crop((left, top, right, bottom))

        im.save("danbooru2020/anime_%s.jpg" % name)



# create_white_noise_dataset(250)

# create_anime_dataset(250)