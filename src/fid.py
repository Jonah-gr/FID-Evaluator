import pickle
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm


def calculate_fid(real_features, generated_features):
    """
    Calculate the Fréchet Inception Distance (FID) between real and generated features.

    Args:
        real_features (numpy.ndarray): Features from real images.
        generated_features (numpy.ndarray): Features from generated images.

    Returns:
        float: The FID score.
    """
    mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = np.mean(generated_features, axis=0), np.cov(generated_features, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def get_fid_scores():
    """
    Load features from a pickle file and calculate FID scores for various noise types and levels.
    Plot the percentage increase in FID scores.

    Returns:
        None
    """
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
    """
    Calculate the percentage increase from the initial value for a list of values.

    Args:
        values (list of float): List of values to calculate the percentage increase.

    Returns:
        list of float: List of percentage increases.
    """
    initial_value = abs(values[0])
    percentage_increases = [0.0]
    for value in values[1:]:
        increase = ((value - initial_value) / initial_value) * 100
        percentage_increases.append(increase)
    return percentage_increases


def plot_percentage_increases(noise_levels, data_dict):
    """
    Plot the percentage increases for different noise types and PCA components.

    Args:
        noise_levels (iterable): X-axis values (noise levels).
        data_dict (dict): Dictionary containing FID scores for different noise types and PCA components.

    Returns:
        None
    """
    num_classes = len(data_dict)
    num_cols = 2
    num_rows = num_classes

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    axes = axes.reshape(num_rows, num_cols)

    for i, (noise_type, noise_data) in enumerate(data_dict.items()):
        ax1 = axes[i, 0]
        for n_components, fid_scores in noise_data.items():
            y_values_noise = calculate_percentage_increases(fid_scores)
            ax1.plot(noise_levels, y_values_noise, marker="o", label=f"{n_components} components")
        ax1.set_ylabel(f"{noise_type}", rotation="horizontal", ha="right")
        ax1.grid(True)

        ax2 = axes[i, 1]
        x_values_components = []
        y_values_components = []
        for n_components, fid_scores in sorted(noise_data.items()):
            y_values_components.append(np.mean(calculate_percentage_increases(fid_scores)[1:]))
            x_values_components.append(n_components)
        ax2.plot(x_values_components, y_values_components, marker="o")
        ax2.grid(True)
    axes[0, 0].legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    import cProfile
    import pstats
    import io

    pr = cProfile.Profile()
    pr.enable()

    my_result = get_fid_scores()

    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
    ps.print_stats()

    with open("cProfile.txt", "w+") as f:
        f.write(s.getvalue())
