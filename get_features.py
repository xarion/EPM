from dataset import get_validation_dataset, get_train_dataset
from features import save_features


def main():
    save_features(get_validation_dataset(), "validation_features")
    save_features(get_train_dataset(), "training_features")


if __name__ == "__main__":
    main()
