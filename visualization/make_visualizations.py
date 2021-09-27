import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import visualization.viz_utils as viz_utils
import pandas as pd
import seaborn as sns
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})


def create_dummy_data():
    rs = np.random.RandomState(seed=0)
    distr_1 = viz_utils.normalize_array_between(rs.rand(50), 0, 1)
    distr_2 = viz_utils.normalize_array_between(rs.rand(50), 1.5, 3)
    distr_3 = viz_utils.normalize_array_between(rs.rand(50), 2, 5)

    return distr_1, distr_2, distr_3


def read_data(path, extra=0):
    data = np.load(path)

    if extra:
        mu = np.mean(data, axis=0)
        cov_mat = np.cov(np.transpose(data))
    else:
        mu = None
        cov_mat = None
    return data, mu, cov_mat


# for now, use this until we have the training data encodings
def create_gaussian_data(extra=0):
    rs = np.random.RandomState(seed=0)
    data = rs.normal(size=(50, 2048))
    data = viz_utils.normalize_array_between(data, data.min(), data.max(), 0, 1)
    if extra:
        mu = np.mean(data, axis=0)
        cov_mat = np.cov(np.transpose(data))
    else:
        mu = None
        cov_mat = None
    return data, mu, cov_mat


def create_density_plot(data, density_labels, name):

    number_of_densities = len(data)

    assert(number_of_densities == len(density_labels)), "The number of densities is not equal to the number of labels."

    data_cat = np.concatenate(data)

    density_labels_rep = np.repeat(density_labels, len(data[0]))

    data_frame = pd.DataFrame(dict(x=data_cat, g=density_labels_rep))

    palette = sns.cubehelix_palette(2*number_of_densities, rot=-.25, light=.7)
    facet_grid = sns.FacetGrid(data_frame, row="g", hue="g", aspect=10, height=0.9, palette=palette)

    # Draw the densities in a few steps
    facet_grid.map(sns.kdeplot, "x", bw_adjust=.5, clip_on=False, fill=True, alpha=1, linewidth=1.5)
    facet_grid.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw_adjust=.5)

    # passing color=None to refline() uses the hue mapping
    facet_grid.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color, ha="left", va="center", transform=ax.transAxes)


    facet_grid.map(label, "x")
    
    # Set the subplots to overlap
    facet_grid.figure.subplots_adjust(hspace=-.25)
    
    # Remove axes details that don't play well with overlap
    facet_grid.set_titles("")
    facet_grid.set(yticks=[], ylabel="", xlabel="Normalized Mahalanobis Distance")

    facet_grid.despine(bottom=True, left=True)


    viz_utils.save_figure(facet_grid, "%s.png" % name, "results")


val_path = "/home/gabi/PycharmProjects/EPM/resnet50_validation_features.npy"
val_encodings = read_data(val_path, extra=0)[0]
gaussian_data = create_gaussian_data(extra=0)[0]

print("gauss-gauss")
# distance_1 = viz_utils.mahalanobis_distance(gaussian_data, gaussian_data)
print("gauss-val")
distance_2 = viz_utils.mahalanobis_distance(gaussian_data, val_encodings)
print("val-val")
distance_3 = viz_utils.mahalanobis_distance(val_encodings, val_encodings)

data = (distance_2, distance_3)
# low, high = viz_utils.get_low_high(data)
# norm_distance_1 = viz_utils.normalize_array_between(distance_1, low, high, 0, 10)
# norm_distance_2 = viz_utils.normalize_array_between(distance_2, low, high, 0, 10)

# data = (norm_distance_1, norm_distance_2)
labels = ["gauss-val", "val-val"]
name = "gaussian_validation"

# create_density_plot((distance_1, distance_2, distance_3), ["gauss-gauss", "gauss-val", "val-val"], name)
create_density_plot(data, labels, name)