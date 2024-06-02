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
    with open("features.pkl", 'rb') as file: 
       features = pickle.load(file) 

    fid_scores = defaultdict(list)

    for noise_level in tqdm(features["real"]["no pca"].keys()):
        fid_scores[2048].append(calculate_fid(features["real"]["no pca"][0.0], features["fake"]["no pca"][noise_level]))
        for n_components in features["real"]["pca"].keys():
            fid_scores[n_components].append(calculate_fid(features["real"]["pca"][n_components][0.0], features["fake"]["pca"][n_components][noise_level]))

    print(fid_scores)
    plot_percentage_increases(features["real"]["no pca"].keys(), fid_scores)

def calculate_percentage_increases(values):
    initial_value = values[0]
    percentage_increases = [0.0]
    for value in values[1:]:
        increase = ((value - initial_value) / initial_value) * 100
        percentage_increases.append(increase)
    return percentage_increases

def plot_percentage_increases(x_values, data_dict):
    plt.figure(figsize=(10, 5))
    
    for key, value_list in data_dict.items():
        y_values = calculate_percentage_increases(value_list)
        plt.plot(x_values, y_values, marker='o', label=key)
    
    plt.xlabel('X values')
    plt.ylabel('Percentage Increase')
    plt.title('Percentage Increase from Initial Value')
    plt.legend()
    plt.grid(True)
    plt.show()
