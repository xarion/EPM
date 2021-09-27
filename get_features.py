import numpy as np
from dataset import get_validation_dataset
from features import features

validation_features = features("resnet50", get_validation_dataset())
np.save("resnet_50_validation_features.npy", validation_features)