import pickle
from collections import defaultdict
from tqdm import tqdm
from sklearn.decomposition import PCA


def nested_defaultdict():
    """
    Create a nested defaultdict where each dictionary will create another nested defaultdict by default.

    Returns:
        defaultdict: A nested defaultdict.
    """
    return defaultdict(nested_defaultdict)


def convert_to_nested_defaultdict(d):
    """
    Recursively convert a standard dictionary to a nested defaultdict.

    Args:
        d (dict): The dictionary to convert.

    Returns:
        defaultdict: A nested defaultdict with the same data as the input dictionary.
    """
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
    """
    Load features from a pickle file and convert them to a nested defaultdict.

    Returns:
        defaultdict: A nested defaultdict containing the loaded features.
    """
    with open("features.pkl", "rb") as file:
        features = pickle.load(file)
    return convert_to_nested_defaultdict(features)


def run_pca(args):
    """
    Perform PCA on the features and save the transformed features back to the pickle file.

    Args:
        args: Arguments containing the number of components for PCA and other parameters.

    Returns:
        None
    """
    features = load_features()
    real_features = features["real"]
    fake_features = features["fake"]

    args.n_components = args.n_components.split()
    args.n_components = [int(num) for num in args.n_components]

    noise_types = fake_features["no pca"].keys()
    noise_levels = fake_features["no pca"][list(noise_types)[0]].keys()

    for n_components in args.n_components:
        pca = PCA(n_components=n_components)
        real_features["pca"][n_components] = pca.fit_transform(real_features["no pca"])

        for noise_type in tqdm(noise_types, desc=f"n_components: {n_components}"):
            for noise_level in noise_levels:
                fake_features["pca"][n_components][noise_type][noise_level] = pca.transform(
                    fake_features["no pca"][noise_type][noise_level]
                )

    with open("features.pkl", "wb") as f:
        pickle.dump(features, f)
        print("All features saved to features.pkl")
