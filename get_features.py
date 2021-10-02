from config import tennis_ball_class
from dataset import get_validation_dataset, get_train_dataset
from features import save_features


def main(model_name, image_class):
    save_features(model_name, image_class, get_validation_dataset(image_class), "validation_features")
    save_features(model_name, image_class, get_train_dataset(image_class), "training_features")


if __name__ == "__main__":
    main("resnet50", tennis_ball_class)
