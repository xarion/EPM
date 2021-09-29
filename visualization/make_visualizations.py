import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import visualization.viz_utils as viz_utils
import pandas as pd
import seaborn as sns
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})


def create_dummy_data(n):
    distributions = []
    rs = np.random.RandomState(seed=0)
    ranges = [[0, 1], [1.5, 3], [2, 5]]
    for i in range(n):
        distr = rs.rand(50)
        distr = viz_utils.normalize_array_between(distr, distr.min(), distr.max(), ranges[i][0], ranges[i][1])
        distributions.append(distr)
    return distributions


def read_data(path):
    data = np.load(path)
    return data


# for now, use this until we have the training data encodings
def create_gaussian_data():
    rs = np.random.RandomState(seed=0)
    data = rs.normal(size=(50, 2048))
    data = viz_utils.normalize_array_between(data, data.min(), data.max(), 0, 1)
    return data


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
    train_encodings = read_data(train_path)

    lime_path_1 = "/home/gabi/PycharmProjects/EPM/xai_methods/lime/lime_0.npy"
    lime_encodings_1 = read_data(lime_path_1)

    lime_path_2 = "/home/gabi/PycharmProjects/EPM/xai_methods/lime/lime_1.npy"
    lime_encodings_2 = read_data(lime_path_2)

    anchor_path_1 = "/home/gabi/PycharmProjects/EPM/xai_methods/anchor/anchor_0.npy"
    anchor_encodings_1 = read_data(anchor_path_1)

    anchor_path_2 = "/home/gabi/PycharmProjects/EPM/xai_methods/anchor/anchor_1.npy"
    anchor_encodings_2 = read_data(anchor_path_2)

    val_path = "/home/gabi/PycharmProjects/EPM/resnet50_validation_features.npy"
    val_encodings = read_data(val_path)

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
    train_encodings = read_data(train_path)

    val_path = "/home/gabi/PycharmProjects/EPM/resnet50_validation_features.npy"
    val_encodings = read_data(val_path)

    lime_path_1 = "/home/gabi/PycharmProjects/EPM/xai_methods/lime/lime_0.npy"
    lime_encodings_1 = read_data(lime_path_1)

    lime_path_2 = "/home/gabi/PycharmProjects/EPM/xai_methods/lime/lime_1.npy"
    lime_encodings_2 = read_data(lime_path_2)

    anchor_path_1 = "/home/gabi/PycharmProjects/EPM/xai_methods/anchor/anchor_0.npy"
    anchor_encodings_1 = read_data(anchor_path_1)

    anchor_path_2 = "/home/gabi/PycharmProjects/EPM/xai_methods/anchor/anchor_1.npy"
    anchor_encodings_2 = read_data(anchor_path_2)

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
    option = 3

    if option == 1:
        val_path = "/home/gabi/PycharmProjects/EPM/resnet50_validation_features.npy"
        val_encodings = read_data(val_path)
        data = viz_utils.mahalanobis_distance(val_encodings, val_encodings)

        df = pd.DataFrame(dict(x=data))
        facet_grid = sns.FacetGrid(df, aspect=2.5, height=3)

        facet_grid.map(sns.kdeplot, "x").set(xscale="log")
        facet_grid.set(xlim=(40, 60))

        facet_grid.savefig("testing.png")

    elif option == 2:
        train_path = "/home/gabi/PycharmProjects/EPM/resnet50_train_features.npy"
        train_encodings = read_data(train_path)
        train = viz_utils.mahalanobis_distance(train_encodings, train_encodings)

        val_path = "/home/gabi/PycharmProjects/EPM/resnet50_validation_features.npy"
        val_encodings = read_data(val_path)
        val = viz_utils.mahalanobis_distance(train_encodings, val_encodings)

        lime_path_1 = "/home/gabi/PycharmProjects/EPM/xai_methods/lime/lime_0.npy"
        lime_encodings_1 = read_data(lime_path_1)
        lime1 = viz_utils.mahalanobis_distance(train_encodings, lime_encodings_1)

        data_cat = np.concatenate([train, val, lime1])
        # print(viz_utils.get_low_high(data_cat))
        labels_train = np.array(["train"]).repeat(len(train))
        labels_val = np.array(["val"]).repeat(len(val))
        labels_lime1 = np.array(["lime1"]).repeat(len(lime1))
        labels = np.concatenate([labels_train, labels_val, labels_lime1])
        df = pd.DataFrame(dict(x=data_cat, g=labels))

        # facet_grid = sns.FacetGrid(df, aspect=2.5, height=3)
        # palette = sns.cubehelix_palette(2, rot=-.25, light=.7)
        facet_grid = sns.FacetGrid(df, row="g", hue="g", aspect=2, height=2.9)

        # facet_grid.map(sns.kdeplot, "x").set(xscale="log")
        facet_grid.map(sns.kdeplot, "x", bw_adjust=0.3, clip_on=False, fill=True, alpha=1, linewidth=0.1,
                       log_scale=(True, False))
        # facet_grid.map(sns.kdeplot, "x", clip_on=False, color="w", lw=1, bw_adjust=.5)
        facet_grid.refline(y=0, linewidth=1, linestyle="-", color=None, clip_on=False)
        # facet_grid.set(ylim=(0, 50))

        def label(x, color, label):
            ax = plt.gca()
            ax.text(0, .2, label, color=color, ha="left", va="center", transform=ax.transAxes)

        facet_grid.map(label, "x")


        # Set the subplots to overlap
        # facet_grid.figure.subplots_adjust(hspace=-.25)
        # Remove axes details that don't play well with overlap
        facet_grid.set_titles("")
        facet_grid.set(yticks=[], ylabel="", xlabel="Mahalanobis Distance on Log Scale")

        facet_grid.despine(bottom=True, left=True)
        # facet_grid.fig.tight_layout()

        facet_grid.savefig("testing2.png")

    elif option == 3:

        train_encodings, train_distances = viz_utils.get_id_data("/home/gabi/PycharmProjects/EPM/resnet50_train_features.npy")
        val_distances = viz_utils.get_data_to_be_measured("/home/gabi/PycharmProjects/EPM/resnet50_validation_features.npy", train_encodings)
        lime_distances = viz_utils.get_data_to_be_measured("/home/gabi/PycharmProjects/EPM/xai_methods/lime", train_encodings)
        anchor_distances = viz_utils.get_data_to_be_measured("/home/gabi/PycharmProjects/EPM/xai_methods/anchor", train_encodings)

        all_distances = [train_distances, val_distances, lime_distances, anchor_distances]

        low, high = viz_utils.get_low_high(all_distances)

        train_distances = viz_utils.normalize_array_between(train_distances, low, high, 0, 1)
        val_distances = viz_utils.normalize_array_between(val_distances, low, high, 0, 1)
        lime_distances = viz_utils.normalize_array_between(lime_distances, low, high, 0, 1)
        anchor_distances = viz_utils.normalize_array_between(anchor_distances, low, high, 0, 1)

        df_train = pd.DataFrame(dict(x=train_distances))
        df_val = pd.DataFrame(dict(x=val_distances))
        df_lime = pd.DataFrame(dict(x=lime_distances))
        df_anchor = pd.DataFrame(dict(x=anchor_distances))
        
        fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(7, 7))

        dfs = [df_train, df_val, df_anchor, df_lime]
        labels = ["train (250)", "val (50)", "anchor (2)", "lime (2)"]

        for i, ax in enumerate(axs):
            sns.kdeplot(data=dfs[i], ax=axs[i], bw_adjust=0.5, fill=True, alpha=1, linewidth=1) #), log_scale=(True, False))

            ax.get_legend().remove()
            ax.grid(True, color='b', linestyle='-', linewidth=0.1, which="major")

            ax.set_ylabel(labels[i])

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_color("b")
            if i < len(labels)-1:
                for tick in ax.xaxis.get_major_ticks():
                    tick.tick1line.set_visible(False)
                    tick.tick2line.set_visible(False)
                    tick.label1.set_visible(False)
                    tick.label2.set_visible(False)

            else:
                ax.set_xlabel("Normalized Mahalanobis distance")


            ax.get_yaxis().set_ticks([])


        xlim = (0, 1)
        plt.setp(axs, xlim=xlim)


        plt.savefig("visualization/scratch/testing3.png")



simple_kde_plot()