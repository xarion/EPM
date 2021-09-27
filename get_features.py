import numpy as np

from config import MODEL_NAME
from dataset import get_validation_dataset
from features import features

validation_features = features(get_validation_dataset())
np.save(f"{MODEL_NAME}_validation_features.npy", validation_features)