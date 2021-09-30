from xai_methods import anchor_lime, feature_permutation, kernelshap, lime, occlusion, shapley_value_sampling


def main():
    feature_permutation.run()
    kernelshap.run()
    lime.run()
    occlusion.run()
    shapley_value_sampling.run()
    anchor_lime.run()


if __name__ == "__main__":
    main()
