import pickle
from collections import defaultdict
from unittest.mock import patch, mock_open
from src.pca import nested_defaultdict, convert_to_nested_defaultdict, load_features, get_n_components, run_pca


def test_nested_defaultdict():
    d = nested_defaultdict()
    d["a"]["b"] = 1
    assert isinstance(d, defaultdict)
    assert isinstance(d["a"], defaultdict)
    assert d["a"]["b"] == 1


def test_convert_to_nested_defaultdict():
    d = {"a": {"b": 1}}
    new_d = convert_to_nested_defaultdict(d)
    assert isinstance(new_d, defaultdict)
    assert isinstance(new_d["a"], defaultdict)
    assert new_d["a"]["b"] == 1


def test_load_features(tmp_path):
    data = {"a": {"b": 1}}
    pkl_file = tmp_path / "test.pkl"
    with open(pkl_file, "wb") as f:
        pickle.dump(data, f)

    loaded_data = load_features(pkl_file)
    assert isinstance(loaded_data, defaultdict)
    assert loaded_data["a"]["b"] == 1


def test_get_n_components():
    input_string = "1, 2, range(3, 5)"
    expected_output = [1, 2, 3, 4]
    assert get_n_components(input_string) == expected_output


class Args:
    def __init__(self, pkl_file, n_components):
        self.pkl_file = pkl_file
        self.n_components = n_components


@patch("src.pca.PCA")
@patch("src.pca.load_features")
@patch("builtins.open", new_callable=mock_open)
@patch("src.pca.pickle.dump")
def test_run_pca(mock_pickle_dump, mock_open_func, mock_load_features, mock_pca, tmp_path):
    # Prepare fake data
    fake_features = {
        "real": {"no pca": [[1, 2], [3, 4]], "pca": {}},
        "fake": {"no pca": {"noise_type1": {"level1": [[5, 6], [7, 8]]}}, "pca": defaultdict(nested_defaultdict)},
    }
    mock_load_features.return_value = convert_to_nested_defaultdict(fake_features)
    args = Args(pkl_file=tmp_path / "test.pkl", n_components="1, 2")

    # Run the function
    run_pca(args)

    # Assertions
    mock_load_features.assert_called_once_with(args.pkl_file)
    assert mock_pca.call_count == 2  # since there are 2 components: 1 and 2
    mock_pickle_dump.assert_called_once()
    with mock_open_func(args.pkl_file, "wb") as f:
        mock_pickle_dump.assert_called_once_with(mock_load_features.return_value, f)
