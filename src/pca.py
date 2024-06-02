import pickle
from collections import defaultdict
from tqdm import tqdm
from sklearn.decomposition import PCA

def nested_defaultdict():
    return defaultdict(nested_defaultdict)

def convert_to_nested_defaultdict(d):
    if isinstance(d, dict):
        new_dict = defaultdict(nested_defaultdict)
        for k, v in d.items():
            if type(k) is list:
                new_dict[k] = v
            else:
                new_dict[k] = convert_to_nested_defaultdict(v)
        return new_dict
    else:
        return d
    
def load_features():
    with open('features.pkl', 'rb') as file:
        features = pickle.load(file)
    return convert_to_nested_defaultdict(features)


def run_pca(args):
    features = load_features()
    real_features = features["real"]
    fake_features = features["fake"]

    args.n_components = args.n_components.split()
    args.n_components = [int(num) for num in args.n_components]

    for n_components in args.n_components:
        pca = PCA(n_components=n_components)
        pca.fit(real_features["no pca"][0.0])

        for noise_level in tqdm(real_features["no pca"].keys()):
            real_features["pca"][n_components][noise_level] = pca.transform(real_features["no pca"][noise_level])
            fake_features["pca"][n_components][noise_level] = pca.transform(fake_features["no pca"][noise_level])
        
    with open("features.pkl", 'wb') as f:
        pickle.dump(features, f)
        print("All features saved to features.pkl")