import pickle
from tqdm import tqdm
from copy import deepcopy
from sklearn.decomposition import PCA
from features import inception

def load_features():
    with open('features.pkl', 'rb') as file: 
        features = pickle.load(file) 
    return features


def run_pca(args):
    real_features = load_features()
    real_features_pca = deepcopy(real_features)

    fake_features = inception(args.path, args.device, noise_levels=real_features["real"].keys(), real=False)
    fake_features_pca = deepcopy(fake_features)

    pca = PCA(n_components=args.n_components)

    pca.fit(real_features["real"][0.0])

    for noise_level in tqdm(real_features["real"].keys()):
        real_features_pca["real"][noise_level] = pca.transform(real_features_pca["real"][noise_level])
        fake_features_pca["fake"][noise_level] = pca.transform(fake_features_pca["fake"][noise_level])
    
    with open("all_features.pkl", 'wb') as f:
        pickle.dump({"real": {"no pca": real_features["real"], "pca": real_features_pca["real"]}, "fake": {"no pca": fake_features["fake"], "pca": fake_features_pca["fake"]}}, f)
        print("All features saved to all_features.pkl")

