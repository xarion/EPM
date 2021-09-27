from dataset import get_validation_dataset
from models import create_feature_extractor_model


def features(dataset):
    model = create_feature_extractor_model()
    return model.predict(dataset)

