import pytest
import torch
from torchvision import transforms
from src.features import (
    add_noise,
    apply_gaussian_blur,
    apply_black_rectangles,
    apply_swirl,
    apply_salt_and_pepper,
    apply_mix_noise,
    load_and_preprocess_image,
    get_noise_types,
    get_noise_levels,
)


@pytest.fixture
def sample_image_tensor():
    # Create a sample image tensor for testing
    return torch.rand((3, 256, 256))


def test_add_noise_gauss(sample_image_tensor):
    noisy_image = add_noise(sample_image_tensor, "gauss", 0.5)
    assert noisy_image.shape == sample_image_tensor.shape


def test_add_noise_blur(sample_image_tensor):
    noisy_image = add_noise(sample_image_tensor, "blur", 0.5)
    assert noisy_image.shape == sample_image_tensor.shape


def test_add_noise_rectangles(sample_image_tensor):
    noisy_image = add_noise(sample_image_tensor, "rectangles", 0.5)
    assert noisy_image.shape == sample_image_tensor.shape


def test_add_noise_swirl(sample_image_tensor):
    noisy_image = add_noise(sample_image_tensor, "swirl", 0.5)
    assert noisy_image.shape == sample_image_tensor.shape


def test_add_noise_salt_and_pepper(sample_image_tensor):
    noisy_image = add_noise(sample_image_tensor, "salt_and_pepper", 0.5)
    assert noisy_image.shape == sample_image_tensor.shape


def test_add_noise_mix(sample_image_tensor):
    noisy_image = add_noise(sample_image_tensor, "mix [gauss blur]", 0.5)
    assert noisy_image.shape == sample_image_tensor.shape


def test_apply_gaussian_blur(sample_image_tensor):
    blurred_image = apply_gaussian_blur(sample_image_tensor, 0.5)
    assert blurred_image.shape == sample_image_tensor.shape


def test_apply_black_rectangles(sample_image_tensor):
    rect_image = apply_black_rectangles(sample_image_tensor, 0.5)
    assert rect_image.shape == sample_image_tensor.shape


def test_apply_swirl(sample_image_tensor):
    swirled_image = apply_swirl(sample_image_tensor, 0.5)
    assert swirled_image.shape == sample_image_tensor.shape


def test_apply_salt_and_pepper(sample_image_tensor):
    sp_image = apply_salt_and_pepper(sample_image_tensor, 0.5)
    assert sp_image.shape == sample_image_tensor.shape


def test_apply_mix_noise(sample_image_tensor):
    mix_image = apply_mix_noise(sample_image_tensor, ["gauss", "blur"], 0.5)
    assert mix_image.shape == sample_image_tensor.shape


def test_load_and_preprocess_image():
    preprocess = transforms.Compose([transforms.Resize(299), transforms.CenterCrop(299), transforms.ToTensor()])
    image_path = "."
    image_generator = load_and_preprocess_image(image_path, preprocess)
    for img in image_generator:
        assert img.shape[1:] == (3, 299, 299)


def test_get_noise_types():
    noise_types_string = "gauss blur mix [swirl, rectangles]"
    expected = ["gauss", "blur", "mix [swirl, rectangles]"]
    result = get_noise_types(noise_types_string)
    assert result == expected


def test_get_noise_levels():
    noise_levels_string = "0.1 0.2 (0.3, 0.4)"
    expected = [0.1, 0.2, (0.3, 0.4)]
    result = get_noise_levels(noise_levels_string)
    assert result == expected
