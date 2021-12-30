import numpy as np
import pandas as pd
import pickle5 as pkl


def read(model, class_name, method):
    with open(f"/disks/bigger/xai_methods/distances/dataframes/{class_name}/{model}/{method}.pkl", "rb") as f:
        data = pkl.load(f).values
        return data


def main():
    hermitries = []
    for model in ["densenet121_unnormalized", "mnasnet1.0_unnormalized", "resnet50_unnormalized"]:
        for class_name in ["chocolatesauce", "printer", "tennisball"]:
            validation_distances = read(model, class_name, "validation_features")
            hermitry_threshold = np.percentile(validation_distances, 95)
            for method in ["AnchorLime", "KernelShap", "Lime", "Occlusion"]:
                d = read(model, class_name, method)
                hermits = np.sum(d > hermitry_threshold)
                hermitry = hermits / len(d)
                hermitries.append({"Hermitry": hermitry,
                                   "XAI Method": method,
                                   "Model": model,
                                   "Class Name": class_name})
    df = pd.DataFrame(hermitries)
    df["Model"] = df["Model"].str.replace("_unnormalized", "")
    df["Model"] = df["Model"].str.replace("densenet121", "DenseNet121")
    df["Model"] = df["Model"].str.replace("resnet50", "ResNet50")
    df["Model"] = df["Model"].str.replace("mnasnet1.0", "MnasNet1.0")

    df["Class Name"] = df["Class Name"].str.replace("chocolatesauce", "Chocolate Sauce")
    df["Class Name"] = df["Class Name"].str.replace("tennisball", "Tennis Ball")
    df["Class Name"] = df["Class Name"].str.replace("printer", "Printer")

    occlusion_mean = df[df["Class Name"] == "Printer"]["Hermitry"].mean()
    print(f"mean Printer: {occlusion_mean:.3f}")

    df["Hermitry"] = df["Hermitry"].map(lambda h: f"{h:.3f}")
    df = df.sort_values(["XAI Method", "Model", "Class Name"], ascending=[False, True, True])
    hermits = df[df["Hermitry"].map(float) > 0.3]
    normies = df[df["Hermitry"].map(float) <= 0.3]

    # print(hermits.to_latex(index=False, caption="Hermitries", label="tab:hermitries", escape=False))
    # print(normies.to_latex(index=False, caption="Hermitries", label="tab:hermitries", escape=False))


    return 0


if __name__ == "__main__":
    main()
