import get_features
from xai_methods import anchor_lime, feature_permutation, kernelshap, lime, occlusion, shapley_value_sampling


def main():
    get_features.main()
    kernelshap.run()
    lime.run()
    occlusion.run()
    shapley_value_sampling.run()
    anchor_lime.run()
    feature_permutation.run()


if __name__ == "__main__":
    main()
