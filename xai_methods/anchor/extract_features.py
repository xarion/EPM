from features import features, features_from_modified_inputs
from models import create_feature_extractor_model
from PIL import Image
from glob import glob
import numpy as np

features_from_modified_inputs("anchor", "_0")
features_from_modified_inputs("anchor", "_1")
