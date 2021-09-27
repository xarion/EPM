from dataset import get_validation_dataset
from models import create_feature_extractor_model


def features(model_name, dataset):
    model = create_feature_extractor_model(model_name)
    return model.predict(dataset)

