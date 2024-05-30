import pickle
from tqdm import tqdm
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

def get_fid_scores(args):
    with open("all_features.pkl", 'rb') as file: 
        all_features = pickle.load(file) 

    fid_scores = []
    fid_scores_pca = []

    for noise_level in tqdm(all_features["real"]["no pca"].keys()):
        fid_scores.append(calculate_fid(all_features["real"]["no pca"][0.0], all_features["fake"]["no pca"][noise_level]))
        fid_scores_pca.append(calculate_fid(all_features["real"]["pca"][0.0], all_features["fake"]["pca"][noise_level]))
        
    print(fid_scores)
    print(fid_scores_pca)
    plot_percentage_increases(all_features["real"]["no pca"].keys(), fid_scores, fid_scores_pca)

def calculate_percentage_increases(values):
    initial_value = values[0]
    percentage_increases = [0.0]
    for value in values[1:]:
        increase = ((value - initial_value) / initial_value) * 100
        percentage_increases.append(increase)
    return percentage_increases

def plot_percentage_increases(x_values, list1, list2):
    y_values1 = calculate_percentage_increases(list1)
    y_values2 = calculate_percentage_increases(list2)
    
    plt.figure(figsize=(10, 5))
    
    plt.plot(x_values, y_values1, marker='o', label='List 1')
    plt.plot(x_values, y_values2, marker='o', label='List 2')
    
    plt.xlabel('X values')
    plt.ylabel('Percentage Increase')
    plt.title('Percentage Increase from Initial Value')
    plt.legend()
    plt.grid(True)
    
    plt.show()
