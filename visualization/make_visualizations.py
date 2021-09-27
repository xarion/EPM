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

    low, high = viz_utils.get_low_high(data)

    normalized_data = []
    for d in data:
        normalized_data.append(viz_utils.normalize_array_between(d, low, high, 0, 1))

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


def create_density_plot_single(data, density_label):

    data_frame = pd.DataFrame(dict(x=data, g=density_label))

    palette = sns.cubehelix_palette(2, rot=-.25, light=.7)
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
    viz_utils.save_figure(facet_grid, "%s.png" % density_label, "results")


def create_single_plots():
    train_path = "/home/gabi/PycharmProjects/EPM/resnet50_train_features.npy"
    train_encodings = read_data(train_path, extra=0)[0]

    lime_path_1 = "/home/gabi/PycharmProjects/EPM/xai_methods/lime/lime_0.npy"
    lime_encodings_1 = read_data(lime_path_1, extra=0)[0]

    lime_path_2 = "/home/gabi/PycharmProjects/EPM/xai_methods/lime/lime_1.npy"
    lime_encodings_2 = read_data(lime_path_2, extra=0)[0]

    anchor_path_1 = "/home/gabi/PycharmProjects/EPM/xai_methods/anchor/anchor_0.npy"
    anchor_encodings_1 = read_data(anchor_path_1, extra=0)[0]

    anchor_path_2 = "/home/gabi/PycharmProjects/EPM/xai_methods/anchor/anchor_1.npy"
    anchor_encodings_2 = read_data(anchor_path_2, extra=0)[0]

    val_path = "/home/gabi/PycharmProjects/EPM/resnet50_validation_features.npy"
    val_encodings = read_data(val_path, extra=0)[0]

    distance_1 = viz_utils.mahalanobis_distance(train_encodings, lime_encodings_1)
    distance_2 = viz_utils.mahalanobis_distance(train_encodings, lime_encodings_2)
    distance_3 = viz_utils.mahalanobis_distance(train_encodings, anchor_encodings_1)
    distance_4 = viz_utils.mahalanobis_distance(train_encodings, anchor_encodings_2)
    distance_5 = viz_utils.mahalanobis_distance(train_encodings, val_encodings)
    distance_6 = viz_utils.mahalanobis_distance(train_encodings, train_encodings)

    low, high = viz_utils.get_low_high([distance_1, distance_2, distance_3, distance_4, distance_5])

    distance_1 = viz_utils.normalize_array_between(distance_1, low, high, 0, 1)
    distance_2 = viz_utils.normalize_array_between(distance_2, low, high, 0, 1)
    distance_3 = viz_utils.normalize_array_between(distance_3, low, high, 0, 1)
    distance_4 = viz_utils.normalize_array_between(distance_4, low, high, 0, 1)
    distance_5 = viz_utils.normalize_array_between(distance_5, low, high, 0, 1)
    distance_6 = viz_utils.normalize_array_between(distance_6, low, high, 0, 1)

    create_density_plot_single(distance_1, "lime1")
    create_density_plot_single(distance_2, "lime2")
    create_density_plot_single(distance_3, "anchor1")
    create_density_plot_single(distance_4, "anchor2")
    create_density_plot_single(distance_6, "train")
    create_density_plot_single(distance_5, "val")


def create_joint_plot():
    train_path = "/home/gabi/PycharmProjects/EPM/resnet50_train_features.npy"
    train_encodings = read_data(train_path, extra=0)[0]

    val_path = "/home/gabi/PycharmProjects/EPM/resnet50_validation_features.npy"
    val_encodings = read_data(val_path, extra=0)[0]

    lime_path_1 = "/home/gabi/PycharmProjects/EPM/xai_methods/lime/lime_0.npy"
    lime_encodings_1 = read_data(lime_path_1, extra=0)[0]

    lime_path_2 = "/home/gabi/PycharmProjects/EPM/xai_methods/lime/lime_1.npy"
    lime_encodings_2 = read_data(lime_path_2, extra=0)[0]

    anchor_path_1 = "/home/gabi/PycharmProjects/EPM/xai_methods/anchor/anchor_0.npy"
    anchor_encodings_1 = read_data(anchor_path_1, extra=0)[0]

    anchor_path_2 = "/home/gabi/PycharmProjects/EPM/xai_methods/anchor/anchor_1.npy"
    anchor_encodings_2 = read_data(anchor_path_2, extra=0)[0]

    distance_1 = viz_utils.mahalanobis_distance(train_encodings, lime_encodings_1)
    distance_2 = viz_utils.mahalanobis_distance(train_encodings, lime_encodings_2)
    distance_3 = viz_utils.mahalanobis_distance(train_encodings, anchor_encodings_1)
    distance_4 = viz_utils.mahalanobis_distance(train_encodings, anchor_encodings_2)
    distance_5 = viz_utils.mahalanobis_distance(train_encodings, val_encodings)
    distance_6 = viz_utils.mahalanobis_distance(train_encodings, train_encodings)

    data = [distance_6, distance_5, distance_1, distance_2, distance_3, distance_4]
    labels = ["train", "val", "lime1", "lime2", "anchor1", "anchor2"]

    create_density_plot(data, labels, "joint")


def simple_kde_plot():
    option = 2
    if option == 1:
        val_path = "/home/gabi/PycharmProjects/EPM/resnet50_validation_features.npy"
        val_encodings = read_data(val_path, extra=0)[0]
        data = viz_utils.mahalanobis_distance(val_encodings, val_encodings)

        df = pd.DataFrame(dict(x=data))
        facet_grid = sns.FacetGrid(df, aspect=2.5, height=3)

        facet_grid.map(sns.kdeplot, "x").set(xscale="log")
        facet_grid.set(xlim=(40, 60))

        facet_grid.savefig("testing.png")

    elif option == 2:
        train_path = "/home/gabi/PycharmProjects/EPM/resnet50_train_features.npy"
        train_encodings = read_data(train_path, extra=0)[0]
        train = viz_utils.mahalanobis_distance(train_encodings, train_encodings)

        val_path = "/home/gabi/PycharmProjects/EPM/resnet50_validation_features.npy"
        val_encodings = read_data(val_path, extra=0)[0]
        val = viz_utils.mahalanobis_distance(train_encodings, val_encodings)

        data_cat = np.concatenate([train, val])
        labels_train = np.array(["train"]).repeat(len(train))
        labels_val = np.array(["val"]).repeat(len(val))
        labels = np.concatenate([labels_train, labels_val])
        df = pd.DataFrame(dict(x=data_cat, g=labels))

        # facet_grid = sns.FacetGrid(df, aspect=2.5, height=3)
        palette = sns.cubehelix_palette(2, rot=-.25, light=.7)
        facet_grid = sns.FacetGrid(df, row="g", hue="g", aspect=2.5, height=3, palette=palette)

        facet_grid.map(sns.kdeplot, "x").set(xscale="log")
        facet_grid.map(sns.kdeplot, "x", bw_adjust=.5, clip_on=False, fill=True, alpha=1, linewidth=1.5)
        facet_grid.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw_adjust=.5)
        facet_grid.refline(y=0, linewidth=1, linestyle="-", color=None, clip_on=False)

        def label(x, color, label):
            ax = plt.gca()
            ax.text(0, .2, label, color=color, ha="left", va="center", transform=ax.transAxes)

        facet_grid.map(label, "x")

        # Set the subplots to overlap
        # facet_grid.figure.subplots_adjust(hspace=-.25)
        # Remove axes details that don't play well with overlap
        facet_grid.set_titles("")
        facet_grid.set(yticks=[], ylabel="", xlabel="Normalized Mahalanobis Distance")

        facet_grid.despine(bottom=True, left=True)
        facet_grid.savefig("testing2.png")


simple_kde_plot()