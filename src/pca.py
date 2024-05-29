import os
import pickle
from sklearn.decomposition import PCA
from features import inception

def load_features():
    with open('features.pkl', 'rb') as file: 
        features = pickle.load(file) 
    return features


def run_pca(args):
    real_features = load_features()

    pca = PCA(n_components=args.n_components)
    real_features_pca = pca.fit_transform(real_features)

    fake_features = inception(args.path, args.device)
    fake_features_pca = pca.transform(fake_features)

    with open("all_features.pkl", 'wb') as f:
        pickle.dump((real_features, real_features_pca, fake_features, fake_features_pca), f)
        print("All features saved to all_features.pkl")

