import re
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


def load_features(pkl_file):
    """
    Load features from a pickle file and convert them to a nested defaultdict.

    Returns:
        defaultdict: A nested defaultdict containing the loaded features.
    """
    with open(pkl_file, "rb") as file:
        features = pickle.load(file)
    return convert_to_nested_defaultdict(features)


def get_n_components(n_components_string):
    """
    Parse a string containing a list of integers or ranges and return a list of integers.
    
    Args:
        n_components_string (str): A string containing a list of integers or ranges.
        
    Returns:
        list: A list of integers extracted from the input string.
    """
    pattern = re.compile(r"(\d+|range\(\d+,\s*\d+(?:,\s*\d+)?\))")
    matches = pattern.findall(n_components_string)
    n_components = []
    for match in matches:
        match = match.strip()
        if match.startswith("range"):
            n_components.extend(list(eval(match)))
        else:
            n_components.append(int(match))
    return n_components


def run_pca(args):
    """
    Perform PCA on the features and save the transformed features back to the pickle file.

    Args:
        args: Arguments containing the number of components for PCA and other parameters.

    Returns:
        None
    """
    features = load_features(args.pkl_file)
    real_features = features["real"]
    fake_features = features["fake"]

    args.n_components = get_n_components(args.n_components)

    noise_types = fake_features["no pca"].keys()

    try:
        found_n_components = list(real_features["pca"].keys())
        args.n_components = sorted([n for n in args.n_components if n not in found_n_components])
        if len(found_n_components) > 0:
            print("n_components found: ", found_n_components)
    except Exception as e:
        print(e)

    for n_components in args.n_components:
        pca = PCA(n_components=n_components)
        real_features["pca"][n_components] = pca.fit_transform(real_features["no pca"])

        for noise_type in tqdm(noise_types, desc=f"n_components: {n_components}"):
            for noise_level in fake_features["no pca"][noise_type].keys():
                fake_features["pca"][n_components][noise_type][noise_level] = pca.transform(
                    fake_features["no pca"][noise_type][noise_level]
                )

    with open(args.pkl_file, "wb") as f:
        pickle.dump(features, f)
        print(f"All features saved to {args.pkl_file}")
