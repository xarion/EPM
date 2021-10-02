import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import visualization.viz_utils as viz_utils
import pandas as pd
import seaborn as sns
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
import os
import shutil


TRAINING = "training_features"
VALIDATION = "validation_features"
LIME = "Lime"
ANCHOR = "AnchorLime"
KERNELSHAP = "KernelShap"
SHAPSAMP = "ShapleyValueSampling"
FEATPERM = "FeaturePermutation"
OCCLUSION = "Occlusion"
WHITENOISE = "white_noise"
ANIME = "anime"

encodings_path = "/media/gabi/DATADRIVE1/EPM/files_npy"
dataframes_path = "/media/gabi/DATADRIVE1/EPM/dataframes"


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


def npy_path(dnn_name, the_class, xai_method):
    base_path = os.path.join(encodings_path, the_class)
    path = os.path.join(base_path, "%s_%s.npy" % (dnn_name, xai_method))
    return path



def prepare_data_unnormalized_dataframes(dnn_name, xai_method, the_class, train_encodings):
    print(dnn_name, xai_method)

    if xai_method == WHITENOISE:
        df_path = os.path.join("/home/gabi/PycharmProjects/EPM/visualization/white_noise_images", "%s_%s.pkl" % (dnn_name, WHITENOISE))
        if not os.path.exists(df_path):
            source_npy = os.path.join("/home/gabi/PycharmProjects/EPM/visualization/white_noise_images", "%s_%s.npy" % (dnn_name, WHITENOISE))
            if os.path.exists(source_npy):
                distances = viz_utils.get_data_to_be_measured(source_npy, train_encodings, dnn_name)
                df = pd.DataFrame(dict(x=distances))
                df.to_pickle(df_path)
            else:
                print("%s does not exist" % source_npy)
                return None
        else:
            df = pd.read_pickle(df_path)
            
    elif xai_method == ANIME:
        df_path = os.path.join("/home/gabi/PycharmProjects/EPM/visualization/danbooru2020", "%s_%s.pkl" % (dnn_name, ANIME))
        if not os.path.exists(df_path):
            source_npy = os.path.join("/home/gabi/PycharmProjects/EPM/visualization/danbooru2020", "%s_%s.npy" % (dnn_name, ANIME))
            if os.path.exists(source_npy):
                distances = viz_utils.get_data_to_be_measured(source_npy, train_encodings, dnn_name)
                df = pd.DataFrame(dict(x=distances))
                df.to_pickle(df_path)
            else:
                print("%s does not exist" % source_npy)
                return None
        else:
            df = pd.read_pickle(df_path)
            
    else:
        df_path = os.path.join(dataframes_path, the_class, dnn_name+"_unnormalized")
        pkl_path = os.path.join(dataframes_path, the_class, dnn_name+"_unnormalized", "%s.pkl" % xai_method)

        if not os.path.exists(df_path):
            print("path does not exist: %s, creating" % df_path)
            os.makedirs(df_path)

        if not os.path.exists(pkl_path):
            source_npy = npy_path(dnn_name, the_class, xai_method)
            if os.path.exists(source_npy):
                distances = viz_utils.get_data_to_be_measured(source_npy, train_encodings, dnn_name)

                df = pd.DataFrame(dict(x=distances))
                df.to_pickle(os.path.join(df_path, "%s.pkl" % xai_method))
            else:
                print("%s does not exist" % source_npy)
                return None

        else:
            df = pd.read_pickle(os.path.join(df_path, "%s.pkl" % xai_method))

    return df


def create_paper_plots(dnn_name, the_class, normalize=True):


    if dnn_name in ["densenet121", "resnet50"]:
        train_encodings = None
        df_train = prepare_data_unnormalized_dataframes(dnn_name, TRAINING, the_class, train_encodings)
    else:
        train_encodings, train_distances = viz_utils.get_id_data(npy_path(dnn_name, the_class, TRAINING), dnn_name)
        df_train = pd.DataFrame(dict(x=train_distances))

    df_val = prepare_data_unnormalized_dataframes(dnn_name, VALIDATION, the_class, train_encodings)
    df_lime = prepare_data_unnormalized_dataframes(dnn_name, LIME, the_class, train_encodings)
    df_anchor = prepare_data_unnormalized_dataframes(dnn_name, ANCHOR, the_class, train_encodings)
    df_kernelshap = prepare_data_unnormalized_dataframes(dnn_name, KERNELSHAP, the_class, train_encodings)
    df_shapsamp = prepare_data_unnormalized_dataframes(dnn_name, SHAPSAMP, the_class, train_encodings)
    df_featperm = prepare_data_unnormalized_dataframes(dnn_name, FEATPERM, the_class, train_encodings)
    df_occlusion = prepare_data_unnormalized_dataframes(dnn_name, OCCLUSION, the_class, train_encodings)
    df_whitenoise = prepare_data_unnormalized_dataframes(dnn_name, WHITENOISE, the_class, train_encodings)
    df_anime = prepare_data_unnormalized_dataframes(dnn_name, ANIME, the_class, train_encodings)

    all_distances = [df_train, df_val, df_lime, df_anchor, df_kernelshap, df_shapsamp, df_featperm, df_occlusion,
                     df_whitenoise, df_anime]

    low, high = viz_utils.get_low_high(all_distances)

    if normalize:

        df_train = viz_utils.normalize_array_between(df_train, low, high, 0, 1)
        df_val = viz_utils.normalize_array_between(df_val, low, high, 0, 1)
        df_lime = viz_utils.normalize_array_between(df_lime, low, high, 0, 1)
        df_anchor = viz_utils.normalize_array_between(df_anchor, low, high, 0, 1)
        df_kernelshap = viz_utils.normalize_array_between(df_kernelshap, low, high, 0, 1)
        df_shapsamp = viz_utils.normalize_array_between(df_shapsamp, low, high, 0, 1)
        df_featperm = viz_utils.normalize_array_between(df_featperm, low, high, 0, 1)
        df_occlusion = viz_utils.normalize_array_between(df_occlusion, low, high, 0, 1)
        df_whitenoise = viz_utils.normalize_array_between(df_whitenoise, low, high, 0, 1)
        df_anime = viz_utils.normalize_array_between(df_anime, low, high, 0, 1)

    dfs = [df_train, df_val, df_lime, df_anchor, df_kernelshap, df_shapsamp, df_featperm, df_occlusion,
           df_whitenoise, df_anime]

    fig, axs = plt.subplots(nrows=len(dfs), ncols=1, figsize=(7,7))

    labels = ["train", "val", "LIME", "anchor LIME", "kernel SHAP", "SHAP val. sampl.", "feature perm.", "occlusion",
              "white noise", "anime"]

    bw = 0.3
    for i, ax in enumerate(axs):
        sns.kdeplot(data=dfs[i], ax=axs[i], bw_adjust=bw, fill=True, alpha=1, linewidth=1) #), log_scale=(True, False))

        ax.get_legend().remove()
        ax.grid(True, color="steelblue", linestyle='--', linewidth=0.1, which="major", alpha=0.4)

        ax.set_ylabel(labels[i], rotation=0, labelpad=5, ha="right", va="top", size=9, color="steelblue",
                      family="sans-serif", weight="bold")

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color("steelblue")

        if i < len(labels)-1:
            for tick in ax.xaxis.get_major_ticks():
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)

        else:
            ax.set_xlabel("Mahalanobis distance %s" % dnn_name, size=10, color="black", family="sans-serif", weight="normal")
            if normalize:
                ax.set_xlabel("Normalized Mahalanobis distance %s" % dnn_name, size=10, color="black", family="sans-serif", weight="normal")
                ax.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=10, color="black", family="sans-serif", weight="normal")

        # ax.get_yaxis().set_ticks([])

    if normalize:
        xlim = (0, 1)

    else:
        xlim = (0, high)

        # ylim = (0, 7e-5)
        # plt.setp(axs, ylim=ylim)

    plt.setp(axs, xlim=xlim)


    plt.tight_layout()

    if normalize:
        plt.savefig("/home/gabi/PycharmProjects/EPM/visualization/paper_images/%s/%s_normalized.png" % (the_class, dnn_name))
    else:
        plt.savefig("/home/gabi/PycharmProjects/EPM/visualization/paper_images/%s/%s_unnormalized.png" % (the_class, dnn_name))


create_paper_plots("densenet121", "printer", normalize=False)