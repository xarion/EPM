import numpy as np
import visualization.viz_utils as viz_utils
import matplotlib as plt



def create_dummy_data():
    distr_1 = viz_utils.normalize_2d_array_between(np.random.rand(5, 10), 0, 1)
    distr_2 = viz_utils.normalize_2d_array_between(np.random.rand(5, 10), 1.5, 3)
    distr_3 = viz_utils.normalize_2d_array_between(np.random.rand(5, 10), 2, 5)

    return distr_1, distr_2, distr_3


def create_scale():

