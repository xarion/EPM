import os
import numpy as np
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import ShrunkCovariance
from sklearn.preprocessing import StandardScaler
import pandas as pd
from multiprocessing import Pool, Queue


def normalize_array_between(data, old_low, old_high, new_low, new_high):
    # old_range = data.max() - data.min()
    old_range = old_high - old_low
    new_range = new_high - new_low
    # normalize_array = (((data - data.min()) * new_range) / old_range) + new_low
    normalize_array = (((data - old_low) * new_range) / old_range) + new_low

    return normalize_array
        

def is_positive_semi_definite(matrix):
    if np.all(np.linalg.eigvals(matrix) >= 0):
        return True
    else:
        return False


# Erdi's method to get the covariance matrix to be PSD
def mahalanobis_distance(id_data, distribution_to_measure):
    # normalize data such that covarience matrix is PSD
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(id_data)
    scaled_training_features = scaler.transform(id_data)
    clf = ShrunkCovariance(assume_centered=True).fit(scaled_training_features)

    if np.all(id_data == distribution_to_measure):
        distances = clf.mahalanobis(scaled_training_features)
    else:
        distances = clf.mahalanobis(scaler.transform(distribution_to_measure))


    return distances


def get_low_high(arrays):
    if isinstance(arrays[0], pd.DataFrame):
        new_arrays = []
        for arr in arrays:
            new_arrays.append(np.squeeze(arr.to_numpy()))
        arrays = new_arrays

    min = np.infty
    max = -np.infty
    for arr in arrays:
        if arr.min() < min:
            min = arr.min()
        if arr.max() > max:
            max = arr.max()

    return min, max


def get_id_data(path, dnn_name):
    encodings = np.load(path)

    distances = mahalanobis_distance(encodings, encodings)
    return encodings, distances


def save_as_df(things):
    npy_file_path = things[0]
    id_encodings = things[1]
    save_path = things[2]
    print("\nnpy path: %s\nencoding shape: %s\nsave_path: %s\n" % (npy_file_path, id_encodings.shape, save_path))
    encoding = np.load(npy_file_path)
    dist = mahalanobis_distance(id_encodings, encoding)
    df_part = pd.DataFrame(dict(x=dist))
    df_part.to_pickle(save_path)
    print("\n%s saved as df\n" % npy_file_path)


def parallel_save_df(things, number_processes=10):
    func = save_as_df
    pool = Pool(processes=number_processes)
    pool.apply_async(func)
    pool.map(func, things)
    return pool


def get_data_to_be_measured(path, id_encodings, dnn_name, multiple_file_mode="extend", parallel=False):
    assert(multiple_file_mode in ["extend", "average"]), "multiple_file_mode incorrect"
    assert(dnn_name in ["densenet121", "resnet50", "mnasnet1.0"]), "dnn_name incorrect"

    the_class = path.split("/")[-2]
    xai_method = path.split("/")[-1]

    class_num = 0
    if the_class == "tennisball":
        class_num = 852
    elif the_class == "printer":
        class_num = 742
    elif the_class == "chocolatesaus":
        class_num = 960

    # check if multiple files or not.
    # if multiple files, take average over the distances
    if os.path.isdir(path):
        if multiple_file_mode == "average":
            npys = [i for i in os.listdir(path) if i.__contains__(".npy")]
            npys.sort()
            distances = []
            for i in npys:
                encoding = np.load(os.path.join(path, i))[1:]
                distances.append(np.mean(mahalanobis_distance(id_encodings, encoding)))
            distances = np.array(distances)

        elif multiple_file_mode == "extend":
            npys = [i for i in os.listdir(path) if i.__contains__(dnn_name)]
            npys.sort()
            print(len(npys))
            distances = []

            if xai_method in ["ShapleyValueSampling", "FeaturePermutation"]:
                if parallel:
                    df_path = os.path.join("/media/gabi/DATADRIVE1/EPM/dataframes", the_class, "%s_unnormalized" % dnn_name, xai_method)
                    dfs_saved = os.listdir(df_path)
                    set_saved_numbers = set([int(i.split(".pkl")[0].split(".")[-1]) for i in dfs_saved])
                    todo_numbers = list(set([i for i in range(1, 17)]) - set_saved_numbers)
                    if len(todo_numbers) > 0:
                        npy_list_todo = [os.path.join(path, "%s_%d_%s_%d.npy" % (dnn_name, class_num, xai_method, i)) for i in todo_numbers]
                        things = [[npy_list_todo[i], id_encodings, os.path.join(df_path, "%s.pkl" % npy_list_todo[i].split(".npy")[0].split("_")[-1])] for i in range(len(npy_list_todo))]
                        parallel_save_df(things)

                    dfs_saved = os.listdir(df_path)

                    for i in dfs_saved:
                        part_path = os.path.join(df_path, i)
                        df = pd.read_pickle(part_path)
                        df = list(np.squeeze(df.to_numpy()))
                        distances.extend(df)

                else:
                    for i in npys:
                        df_path = os.path.join("/media/gabi/DATADRIVE1/EPM/dataframes", the_class, "%s_unnormalized" % dnn_name, xai_method)
                        number = i.split(".npy")[0].split("_")[-1]
                        part_path = os.path.join(df_path, "%s.pkl" % number)

                        if not os.path.exists(part_path):
                            if not os.path.exists(df_path):
                                os.makedirs(df_path)

                            print(i)
                            encoding = np.load(os.path.join(path, i))
                            dist = mahalanobis_distance(id_encodings, encoding)
                            df_part = pd.DataFrame(dict(x=dist))
                            df_part.to_pickle(part_path)

                            distances.extend(dist)

            else:
                for i in npys:
                    encoding = np.load(os.path.join(path, i))
                    dist = mahalanobis_distance(id_encodings, encoding)
                    distances.extend(dist)

## --------
            # for i in npys:
            #     if xai_method in ["ShapleyValueSampling", "FeaturePermutation"]:
            #         df_path = os.path.join("/media/gabi/DATADRIVE1/EPM/dataframes", the_class, "%s_unnormalized" % dnn_name, xai_method)
            #         number = i.split(".npy")[0].split("_")[-1]
            #         part_path = os.path.join(df_path, "%s.pkl" % number)
            #
            #         if not os.path.exists(part_path):
            #             if not os.path.exists(df_path):
            #                 os.makedirs(df_path)
            #
            #             print(i)
            #             encoding = np.load(os.path.join(path, i))
            #             dist = mahalanobis_distance(id_encodings, encoding)
            #             df_part = pd.DataFrame(dict(x=dist))
            #             df_part.to_pickle(part_path)
            #
            #             distances.extend(dist)
            #
            #         else:
            #             df = pd.read_pickle(part_path)
            #             df = list(np.squeeze(df.to_numpy()))
            #             distances.extend(df)
            #
            #     else:
            #         encoding = np.load(os.path.join(path, i))
            #         dist = mahalanobis_distance(id_encodings, encoding)
            #         distances.extend(dist)

            distances = np.array(distances)

        else:
            print("multiple_file_mode incorrect")
            return None

    else:
        encoding = np.load(path)
        distances = mahalanobis_distance(id_encodings, encoding)

    return distances