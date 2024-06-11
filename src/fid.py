import math
import pickle
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm


def calculate_fid(real_features, generated_features):
    mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = np.mean(generated_features, axis=0), np.cov(generated_features, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def get_fid_scores():
    with open("features.pkl", "rb") as file:
        features = pickle.load(file)
    fid_scores = {}

    noise_types = features["fake"]["no pca"].keys()
    noise_levels = features["fake"]["no pca"][list(noise_types)[0]].keys()

    for noise_type in noise_types:
        fid_scores[noise_type] = defaultdict(list)
        for noise_level in noise_levels:
            fid_scores[noise_type][2048].append(
                calculate_fid(features["real"]["no pca"], features["fake"]["no pca"][noise_type][noise_level])
            )
            for n_components in tqdm(
                features["fake"]["pca"].keys(), desc=f"Noise Type: {noise_type} | Noise Level: {noise_level}"
            ):
                fid_scores[noise_type][n_components].append(
                    calculate_fid(
                        features["real"]["pca"][n_components],
                        features["fake"]["pca"][n_components][noise_type][noise_level],
                    )
                )

    print(fid_scores)
    plot_percentage_increases(noise_levels, fid_scores)


def calculate_percentage_increases(values):
    initial_value = abs(values[0])
    percentage_increases = [0.0]
    for value in values[1:]:
        increase = ((value - initial_value) / initial_value) * 100
        percentage_increases.append(increase)
    return percentage_increases


def plot_percentage_increases(x_values, data_dict):
    num_classes = len(data_dict)
    num_cols = 2
    num_rows = math.ceil(num_classes / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 2.5 * num_rows))
    axes = axes.flatten()

    for ax in axes[num_classes:]:
        ax.axis("off")

    for ax, (noise_type, noise_data) in zip(axes, data_dict.items()):
        for key, value_list in noise_data.items():
            y_values = calculate_percentage_increases(value_list)
            ax.plot(x_values, y_values, marker="o", label=f"{key}")

        ax.set_title(f"Noise Type: {noise_type}")
        ax.set_xlabel("Noise level")
        ax.set_ylabel("FID: Percentage Increase")
        ax.legend()
        ax.grid(True)

    fig.tight_layout()
    plt.show()