import pytest
import numpy as np
from unittest.mock import patch, mock_open, MagicMock
from src.fid import calculate_fid, get_fid_scores, calculate_percentage_increases, plot_percentage_increases


def test_calculate_fid():
    real_features = np.array([[1, 2], [3, 4], [5, 6]])
    generated_features = np.array([[1, 1], [2, 2], [3, 3]])
    fid_score = calculate_fid(real_features, generated_features)
    assert isinstance(fid_score, float)
    assert fid_score >= 0


@patch("src.fid.pickle.load")
@patch("builtins.open", new_callable=mock_open)
@patch("src.fid.calculate_fid")
@patch("src.fid.plot_percentage_increases")
def test_get_fid_scores(mock_plot_percentage_increases, mock_calculate_fid, mock_open_func, mock_pickle_load, tmp_path):
    # Prepare fake data
    fake_features = {
        "real": {"no pca": np.array([[1, 2], [3, 4]]), "pca": {2: np.array([[1, 2], [3, 4]])}},
        "fake": {
            "no pca": {"noise_type1": {"level1": np.array([[5, 6], [7, 8]])}},
            "pca": {2: {"noise_type1": {"level1": np.array([[5, 6], [7, 8]])}}},
        },
    }
    mock_pickle_load.return_value = fake_features
    mock_calculate_fid.return_value = 0.5
    args = MagicMock()
    args.pkl_file = tmp_path / "test.pkl"

    # Run the function
    get_fid_scores(args)

    # Assertions
    mock_open_func.assert_called_once_with(args.pkl_file, "rb")
    mock_pickle_load.assert_called_once()
    mock_calculate_fid.assert_called()
    mock_plot_percentage_increases.assert_called_once()


def test_calculate_percentage_increases():
    values = [1.0, 2.0, 3.0]
    expected_output = [0.0, 100.0, 200.0]
    assert calculate_percentage_increases(values) == expected_output


@patch("src.fid.calculate_percentage_increases")
@patch("src.fid.plt.show")
def test_plot_percentage_increases(mock_plt_show, mock_calculate_percentage_increases):
    noise_levels = {"noise_type1": ["level1", "level2"]}
    data_dict = {"noise_type1": {2: [1.0, 2.0], 3: [1.5, 2.5]}}
    mock_calculate_percentage_increases.side_effect = lambda x: [(v - x[0]) / x[0] * 100 for v in x]

    plot_percentage_increases(noise_levels, data_dict)

    mock_calculate_percentage_increases.assert_called()
    mock_plt_show.assert_called_once()


if __name__ == "__main__":
    pytest.main()
