import get_features
from config import MODEL_NAMES, IMAGE_CLASSES
from xai_methods import anchor_lime, feature_permutation, kernelshap, lime, occlusion, shapley_value_sampling


def main():
    for model_name in MODEL_NAMES:
        for image_class in IMAGE_CLASSES:
            get_features.main(model_name, image_class)
            kernelshap.run(model_name, image_class)
            lime.run(model_name, image_class)
            occlusion.run(model_name, image_class)
            shapley_value_sampling.run(model_name, image_class)
            anchor_lime.run(model_name, image_class)
            feature_permutation.run(model_name, image_class)


if __name__ == "__main__":
    main()
