from xai_methods import anchor, feature_permutation, kernelshap, lime, occlusion, shapley_value_sampling


def main():
    feature_permutation.run()
    kernelshap.run()
    lime.run()
    occlusion.run()
    shapley_value_sampling.run()
    anchor.run()
