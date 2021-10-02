import os
import numpy as np
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import ShrunkCovariance
from sklearn.preprocessing import StandardScaler
import pandas as pd


def normalize_array_between(data, old_low, old_high, new_low, new_high):
    # old_range = data.max() - data.min()
    old_range = old_high - old_low
    new_range = new_high - new_low
    # normalize_array = (((data - data.min()) * new_range) / old_range) + new_low
    normalize_array = (((data - old_low) * new_range) / old_range) + new_low

    return normalize_array


def save_figure(grid, name, base):

    assert(base in ["results", "ood"]), "base can be 'results' or 'ood'"
    pre = ""

    if base == "results":
        pre = "paper_images"
        path = os.path.join(pre, name)
        grid.savefig(path)

    elif base == "ood":
        pre = "ood_images"
        path = os.path.join(pre, name)
        grid.save(path)
        

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


def get_data_to_be_measured(path, id_encodings, dnn_name, multiple_file_mode="extend"):
    assert(multiple_file_mode in ["extend", "average"]), "multiple_file_mode incorrect"
    assert(dnn_name in ["densenet121", "resnet50", "mnasnet1.0"]), "dnn_name incorrect"

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
            distances = []
            for i in npys:
                encoding = np.load(path)
                distances.extend(mahalanobis_distance(id_encodings, encoding))
            distances = np.array(distances)

        else:
            print("multiple_file_mode incorrect")
            return None

    else:
        encoding = np.load(path)
        distances = mahalanobis_distance(id_encodings, encoding)

    return distances
