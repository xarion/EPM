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

    if xai_method in [SHAPSAMP, FEATPERM]:
        path = os.path.join(base_path, xai_method)
    else:
        path = os.path.join(base_path, "%s_%s.npy" % (dnn_name, xai_method))

    return path


def prepare_data_unnormalized_dataframes(dnn_name, xai_method, the_class, train_encodings, parallel):
    print(dnn_name, xai_method)

    if xai_method == WHITENOISE:
        df_path = os.path.join("/home/gabi/PycharmProjects/EPM/visualization/white_noise_images", "%s_%s.pkl" % (dnn_name, WHITENOISE))
        if not os.path.exists(df_path):
            source_npy = os.path.join("/home/gabi/PycharmProjects/EPM/visualization/white_noise_images", "%s_%s.npy" % (dnn_name, WHITENOISE))
            if os.path.exists(source_npy):
                distances = viz_utils.get_data_to_be_measured(source_npy, train_encodings, dnn_name, parallel=parallel)
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
                distances = viz_utils.get_data_to_be_measured(source_npy, train_encodings, dnn_name, parallel=parallel)
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
                distances = viz_utils.get_data_to_be_measured(source_npy, train_encodings, dnn_name, parallel=parallel)

                df = pd.DataFrame(dict(x=distances))
                df.to_pickle(os.path.join(df_path, "%s.pkl" % xai_method))
            else:
                print("%s does not exist" % source_npy)
                return None

        else:
            df = pd.read_pickle(os.path.join(df_path, "%s.pkl" % xai_method))

    return df


def create_paper_plots(dnn_name, the_class, normalize=True, use_white_noise=False, parallel=False, use_all=False, plot_name=None):


    # if dnn_name in ["densenet121", "resnet50"]:
    #     train_encodings = None
    #     df_train = prepare_data_unnormalized_dataframes(dnn_name, TRAINING, the_class, train_encodings)
    # else:
    train_encodings, train_distances = viz_utils.get_id_data(npy_path(dnn_name, the_class, TRAINING), dnn_name)
    df_train = pd.DataFrame(dict(x=train_distances))

    df_shapsamp = None
    df_featperm = None

    df_val = prepare_data_unnormalized_dataframes(dnn_name, VALIDATION, the_class, train_encodings, parallel=parallel)
    df_lime = prepare_data_unnormalized_dataframes(dnn_name, LIME, the_class, train_encodings, parallel=parallel)
    df_anchor = prepare_data_unnormalized_dataframes(dnn_name, ANCHOR, the_class, train_encodings, parallel=parallel)
    df_kernelshap = prepare_data_unnormalized_dataframes(dnn_name, KERNELSHAP, the_class, train_encodings, parallel=parallel)
    if use_all:
        df_shapsamp = prepare_data_unnormalized_dataframes(dnn_name, SHAPSAMP, the_class, train_encodings, parallel=parallel)
        df_featperm = prepare_data_unnormalized_dataframes(dnn_name, FEATPERM, the_class, train_encodings, parallel=parallel)
    df_occlusion = prepare_data_unnormalized_dataframes(dnn_name, OCCLUSION, the_class, train_encodings, parallel=parallel)
    df_whitenoise = prepare_data_unnormalized_dataframes(dnn_name, WHITENOISE, the_class, train_encodings, parallel=parallel)
    df_anime = prepare_data_unnormalized_dataframes(dnn_name, ANIME, the_class, train_encodings, parallel=parallel)

    # all_distances = [df_train, df_val, df_lime, df_anchor, df_kernelshap, df_occlusion, df_anime]
    if use_all:
        all_distances = [df_train, df_val, df_whitenoise, df_anime, df_lime, df_anchor, df_kernelshap, df_shapsamp, df_featperm, df_occlusion]
    else:
        all_distances = [df_train, df_val, df_whitenoise, df_anime, df_lime, df_anchor, df_kernelshap, df_occlusion]

    low, high = viz_utils.get_low_high(all_distances)

    if normalize:

        df_train = viz_utils.normalize_array_between(df_train, low, high, 0, 1)
        df_val = viz_utils.normalize_array_between(df_val, low, high, 0, 1)
        df_lime = viz_utils.normalize_array_between(df_lime, low, high, 0, 1)
        df_anchor = viz_utils.normalize_array_between(df_anchor, low, high, 0, 1)
        df_kernelshap = viz_utils.normalize_array_between(df_kernelshap, low, high, 0, 1)
        if use_all:
            df_shapsamp = viz_utils.normalize_array_between(df_shapsamp, low, high, 0, 1)
            df_featperm = viz_utils.normalize_array_between(df_featperm, low, high, 0, 1)
        df_occlusion = viz_utils.normalize_array_between(df_occlusion, low, high, 0, 1)
        df_whitenoise = viz_utils.normalize_array_between(df_whitenoise, low, high, 0, 1)
        df_anime = viz_utils.normalize_array_between(df_anime, low, high, 0, 1)
    
    if use_all:
        dfs = [df_train, df_val, df_whitenoise, df_anime, df_lime, df_anchor, df_kernelshap, df_shapsamp, df_featperm, df_occlusion]
    else:
        dfs = [df_train, df_val, df_whitenoise, df_anime, df_lime, df_anchor, df_kernelshap, df_occlusion]

    fig, axs = plt.subplots(nrows=len(dfs), ncols=1, figsize=(7,7))

    if use_white_noise:
        if use_all:
            labels = ["train", "val", "white noise", "anime", "LIME", "anchor LIME", "kernel SHAP","Shapley val. samp.", "feature perm.", "occlusion"]
        else:
            labels = ["train", "val", "white noise", "anime", "LIME", "anchor LIME", "kernel SHAP", "occlusion"]
    else:
        labels = ["train", "val", "anime", "LIME", "anchor LIME", "kernel SHAP", "occlusion"]


    fig.suptitle("%s %s" % (viz_utils.get_fancy(dnn_name, the_class)))

    bw = 0.3
    for i, ax in enumerate(axs):
        g = sns.kdeplot(data=dfs[i], ax=axs[i], bw_adjust=bw, fill=True, alpha=1, linewidth=1) #), log_scale=(True, False))

        ax.get_legend().remove()
        ax.grid(True, color="steelblue", linestyle='--', linewidth=0.2, which="major", alpha=1)

        ax.set_ylabel(labels[i], rotation=0, labelpad=5, ha="right", va="top", size=10, color="black",
                      family="sans-serif", weight="normal")

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color("steelblue")
        ax.yaxis.tick_right()

        ylabels = list(g.get_yticks())
        # print(ylabels)
        new_ylabel = []
        for y in ylabels:
            if y != 0.:
                y = "{:e}".format(float(y))
                p1, p2 = y.split(".")
                p2 = str(int(p2.split("e")[-1]))
                if p2 == "0":
                    y = p1
                else:
                    y = p1 + "e" + p2
            else:
                y = "0"
            new_ylabel.append(y)
        ylabels = new_ylabel

        ax.set_yticklabels(ylabels, fontsize=9, color="black", family="sans-serif", weight="normal")
        ax.yaxis.set_tick_params(length=0)


        if i < len(labels)-1:
            for tick in ax.xaxis.get_major_ticks():
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)

        else:
            ax.set_xlabel("Mahalanobis distance", size=10, color="black", family="sans-serif", weight="normal")
            # xlabels = list(g.get_xticks())
            # ax.set_xticklabels(xlabels, fontsize=9, color="black", family="sans-serif", weight="normal")

            if normalize:
                ax.set_xlabel("Normalized Mahalanobis distance", size=10, color="black", family="sans-serif", weight="normal")
                ax.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=9, color="black", family="sans-serif", weight="normal")

        # ax.get_yaxis().set_ticks([])

    if normalize:
        xlim = (0, 1)

    else:
        xlim = (0, high)

        # ylim = (0, 0.0007)
        # plt.setp(axs, ylim=ylim)

    plt.setp(axs, xlim=xlim)


    plt.tight_layout()

    if not plot_name is None:
        plt.savefig(plot_name)
    elif normalize:
        plt.savefig("/home/gabi/PycharmProjects/EPM/visualization/paper_images/%s/%s_normalized.png" % (the_class, dnn_name))
    else:
        if use_white_noise:
            if use_all:
                plt.savefig("/home/gabi/PycharmProjects/EPM/visualization/paper_images/%s/%s_unnormalized_wn.png" % (the_class, dnn_name))
            else:
                plt.savefig("/home/gabi/PycharmProjects/EPM/visualization/paper_images/%s/%s_unnormalized_wn_min2meths.png" % (the_class, dnn_name))
                
        else:
            plt.savefig("/home/gabi/PycharmProjects/EPM/visualization/paper_images/%s/%s_unnormalized.png" % (the_class, dnn_name))
        
        
        
            
    

# fixing the plots
pname = os.path.join("/home/gabi/PycharmProjects/EPM/visualization/scratch", "fixing_plot.png")
create_paper_plots("resnet50", "tennisball", normalize=False, use_white_noise=True, use_all=False, plot_name=pname)


# ---------------------------------------------------------------------------------

# WITH ShapleyValueSampling + FeaturePermutation
# create_paper_plots("mnasnet1.0", "tennisball", normalize=False, use_white_noise=True) # DONE
# create_paper_plots("mnasnet1.0", "printer", normalize=False, use_white_noise=True, parallel=False) # 01:44 # DONE
# create_paper_plots("mnasnet1.0", "chocolatesauce", normalize=False, use_white_noise=True, parallel=False, use_all=True) # DONE
#
# create_paper_plots("densenet121", "tennisball", normalize=False, use_white_noise=True) # DONE
# create_paper_plots("densenet121", "printer", normalize=False, use_white_noise=True, parallel=False, use_all=True)
# create_paper_plots("densenet121", "chocolatesauce", normalize=False, use_white_noise=True, parallel=False, use_all=True)
#
# create_paper_plots("resnet50", "tennisball", normalize=False, use_white_noise=True)
# create_paper_plots("resnet50", "printer", normalize=False, use_white_noise=True, parallel=False)
# create_paper_plots("resnet50", "chocolatesauce", normalize=False, use_white_noise=True, parallel=False)

# ---------------------------------------------------------------------------------

# WITHOUT ShapleyValueSampling + FeaturePermutation
# create_paper_plots("mnasnet1.0", "tennisball", normalize=False, use_white_noise=True, use_all=False) # DONE
# create_paper_plots("mnasnet1.0", "printer", normalize=False, use_white_noise=True, parallel=False, use_all=False) # DONE
# create_paper_plots("mnasnet1.0", "chocolatesauce", normalize=False, use_white_noise=True, parallel=False, use_all=False) # DONE
#
# create_paper_plots("densenet121", "tennisball", normalize=False, use_white_noise=True, use_all=False) # DONE
# create_paper_plots("densenet121", "printer", normalize=False, use_white_noise=True, parallel=False, use_all=False) # DONE
# create_paper_plots("densenet121", "chocolatesauce", normalize=False, use_white_noise=True, parallel=False, use_all=False) # DONE
#
# create_paper_plots("resnet50", "tennisball", normalize=False, use_white_noise=True, use_all=False)
# create_paper_plots("resnet50", "printer", normalize=False, use_white_noise=True, parallel=False, use_all=False)
# create_paper_plots("resnet50", "chocolatesauce", normalize=False, use_white_noise=True, parallel=False, use_all=False)