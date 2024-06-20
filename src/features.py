import os
import re
import torch
import pickle
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torchvision import models, transforms
from torchvision.models.inception import Inception_V3_Weights
from PIL import Image, ImageFilter, ImageDraw
from skimage.transform import swirl


def add_noise(image, noise_type, noise_level):
    """
    Adds specified noise to an image.

    Parameters:
        image (Tensor): The input image tensor.
        noise_level (float): The level of noise to be added.
        noise_type (str): The type of noise to add (e.g., "gauss", "blur", "rectangles", "swirl", "salt_and_pepper").

    Returns:
        Tensor: The noisy image tensor.
    """
    if noise_level == 0.0:
        return image
    if noise_type == "gauss":
        noisy_image = image + noise_level * torch.randn_like(image)
        noisy_image = torch.clamp(noisy_image, 0.0, 1.0)
    elif noise_type == "blur":
        noisy_image = apply_gaussian_blur(image, noise_level)
    elif noise_type == "rectangles":
        noisy_image = apply_black_rectangles(image, noise_level)
    elif noise_type == "swirl":
        noisy_image = apply_swirl(image, noise_level)
    elif noise_type == "salt_and_pepper":
        noisy_image = apply_salt_and_pepper(image, noise_level)
    elif noise_type.startswith("mix"):
        mix_noise_types = noise_type[5:-1].split()
        noisy_image = apply_mix_noise(image, mix_noise_types, noise_level)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    return noisy_image


def apply_gaussian_blur(image, noise_level):
    """
    Applies Gaussian blur to an image.

    Parameters:
        image (Tensor): The input image tensor.
        noise_level (float): The level of blur to apply.

    Returns:
        Tensor: The blurred image tensor.
    """
    pil_image = transforms.ToPILImage()(image).convert("RGB")
    blur_radius_mapped = noise_level * 10
    blurred_image = pil_image.filter(ImageFilter.GaussianBlur(radius=blur_radius_mapped))
    return transforms.ToTensor()(blurred_image)


def apply_black_rectangles(image, noise_level, grid_size=8):
    """
    Applies black rectangles to an image.

    Parameters:
        image (Tensor): The input image tensor.
        noise_level (float): The level of rectangles to apply.
        grid_size (int): The grid size for the rectangles.

    Returns:
        Tensor: The image tensor with black rectangles.
    """
    pil_image = transforms.ToPILImage()(image).convert("RGB")
    draw = ImageDraw.Draw(pil_image)
    width, height = pil_image.size
    cell_width = width // grid_size
    cell_height = height // grid_size

    total_cells = grid_size * grid_size
    cells_to_fill = int(total_cells * noise_level)

    cells = [(i, j) for i in range(grid_size) for j in range(grid_size)]
    np.random.shuffle(cells)

    for cell in cells[:cells_to_fill]:
        x1 = cell[0] * cell_width
        y1 = cell[1] * cell_height
        x2 = x1 + cell_width
        y2 = y1 + cell_height
        draw.rectangle([x1, y1, x2, y2], fill="black")

    return transforms.ToTensor()(pil_image)


def apply_swirl(image, noise_level):
    """
    Applies swirl distortion to an image.

    Parameters:
        image (Tensor): The input image tensor.
        noise_level (float): The level of swirl to apply.

    Returns:
        Tensor: The swirled image tensor.
    """
    pil_image = transforms.ToPILImage()(image).convert("RGB")
    np_image = np.array(pil_image)
    swirled_image = swirl(np_image, strength=noise_level * 15, radius=250)
    swirled_image = (swirled_image * 255).astype(np.uint8)
    return transforms.ToTensor()(Image.fromarray(swirled_image))


def apply_salt_and_pepper(image, noise_level):
    """
    Applies salt-and-pepper noise to an image.

    Parameters:
        image (Tensor): The input image tensor.
        noise_level (float): The level of noise to apply.

    Returns:
        Tensor: The image tensor with salt-and-pepper noise.
    """
    pil_image = transforms.ToPILImage()(image).convert("RGB")
    image_np = np.array(pil_image)

    row, col, ch = image_np.shape
    s_vs_p = 0.5
    amount = noise_level / 10
    out = np.copy(image_np)

    num_salt = np.ceil(amount * image_np.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image_np.shape]
    out[coords[0], coords[1], :] = 255

    num_pepper = np.ceil(amount * image_np.size * (1.0 - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image_np.shape]
    out[coords[0], coords[1], :] = 0

    noisy_image_pil = Image.fromarray(out)
    return transforms.ToTensor()(noisy_image_pil)


def apply_mix_noise(image, mix_noise_types, noise_level):
    """
    Applies a mixture of different types of noise to an image.

    Parameters:
        image (Tensor): The input image tensor.
        mix_noise_types (List[str]): A list of noise types to apply to the image.
        noise_level (float): The level of noise to apply to the image.

    Returns:
        Tensor: The noisy image tensor.
    """
    for noise_type in mix_noise_types:
        image = add_noise(image, noise_type, noise_level)
    return image


def load_and_preprocess_image(img_path, transform):
    """
    Loads and preprocesses an image.

    Parameters:
        img_path (str): The path to the image file.
        transform (Compose): The transformations to apply to the image.

    Returns:
        Tensor: The preprocessed image tensor.
    """
    for root, _, files in os.walk(img_path):
        for file in files:
            if file.lower().endswith(("png", "jpg", "jpeg")):
                file_path = os.path.join(root, file)
                try:
                    image = Image.open(file_path).convert("RGB")
                    image = transform(image)
                    image = image.unsqueeze(0)
                    yield image
                except Exception as e:
                    print(f"Error loading image {file_path}: {e}")


def inception(path, device, noise_levels, noise_types):
    """
    Extracts features from images using the Inception v3 model with optional noise.

    Parameters:
        path (str): The path to the directory containing images.
        device (str): The device to run the model on (cpu or cuda).
        noise_levels (list of float): The levels of noise to apply.
        noise_types (list of str): The types of noise to apply.

    Returns:
        dict: A dictionary containing the extracted features for each noise type and level.
    """
    real_features = True if "no noise" in noise_types else False
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")

    model = models.inception_v3(weights=Inception_V3_Weights.DEFAULT)
    model.fc = nn.Identity()
    model.eval()
    model.to(device)

    preprocess = transforms.Compose([transforms.Resize(299), transforms.CenterCrop(299), transforms.ToTensor()])
    features_dict = {k: {} for k in noise_types}

    with torch.no_grad():
        for noise_type in noise_types:
            for noise_level in noise_levels:
                features = []
                for img in tqdm(
                    load_and_preprocess_image(path, preprocess),
                    desc=f"Noise Type: {noise_type} | Noise Level: {noise_level}",
                    total=len(os.listdir(path)),
                ):
                    img = add_noise(img.squeeze(0), noise_type, noise_level)
                    img = img.unsqueeze(0).to(device)
                    feature = model(img).cpu().numpy().flatten()
                    features.append(feature)
                features_dict[noise_type][noise_level] = features
                with open(f"checkpoints/checkpoint_{'real' if real_features else 'fake'}.pkl", "wb") as f:
                    pickle.dump(features_dict, f)
    if real_features:
        return features_dict["no noise"][0.0]
    return features_dict


def get_noise_types(noise_types_string):
    """
    Extracts noise types from a given noise types string using a regular expression pattern.

    Parameters:
        noise_types_string (str): A string containing noise types.

    Returns:
        List: A list of noise types extracted from the input string.
    """
    pattern = r"\bmix[^\]]*\]|\S+"
    noise_types = re.findall(pattern, noise_types_string)
    return noise_types


def compute_features(args):
    """
    Computes features for real and fake images, optionally adding noise, and saves them to a file.

    Parameters:
        args (Namespace): The command-line arguments containing paths and settings.

    Returns:
        None
    """
    real_features = {}
    fake_features = {}

    args.noise = args.noise.replace(",", "").split()
    args.noise = [float(num) for num in args.noise]
    if 0.0 not in args.noise:
        args.noise.append(0.0)
    args.noise.sort()

    args.noise_types = get_noise_types(args.noise_types.replace(",", ""))
    if "all" in args.noise_types:
        args.noise_types.extend(["gauss", "blur", "swirl", "rectangles", "salt_and_pepper"])
        args.noise_types.remove("all")

    if args.real:
        if not args.fake:
            try:
                with open("features.pkl", "rb") as f:
                    features = pickle.load(f)
                    fake_features = features["fake"]["no pca"]
            except:
                print("No previous fake features found")
        print("Real images:")
        real_features = inception(args.real, args.device, [0.0], ["no noise"])

    if args.fake:
        if not args.real:
            try:
                with open("features.pkl", "rb") as f:
                    features = pickle.load(f)
                    real_features = features["real"]["no pca"]
                    print("Previous real features found")
            except:
                print("No previous real features found")
        print("Fake images:")
        fake_features = inception(args.fake, args.device, args.noise, args.noise_types)

    with open("features.pkl", "wb") as f:
        pickle.dump({"real": {"no pca": real_features}, "fake": {"no pca": fake_features}}, f)
    print("Features saved to features.pkl")


if __name__ == "__main__":
    import requests
    import matplotlib.pyplot as plt

    if not os.path.exists("dog.jpg"):
        url = "https://images.unsplash.com/photo-1624956578877-4948166c5dcb?q=80&w=1974&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
        response = requests.get(url)
        with open("dog.jpg", "wb") as file:
            file.write(response.content)

    preprocess = transforms.Compose([transforms.Resize(299), transforms.CenterCrop(299), transforms.ToTensor()])

    noise_types = [
        "gauss",
        "blur",
        "swirl",
        "rectangles",
        "salt_and_pepper",
        "mix [swirl, rectangles]",
        "mix [rectangles, swirl]",
    ]

    fig, axs = plt.subplots(len(noise_types), 11, figsize=(20, 15))

    image = next(load_and_preprocess_image(".", preprocess))
    for i in range(len(noise_types)):
        for k in np.arange(0.0, 1.1, 0.1):
            img = add_noise(image.squeeze(0), noise_types[i], k)
            img = transforms.ToPILImage()(img).convert("RGB")
            axs[0][int(k * 10)].set_title(f"{np.round(k, 1)}")
            axs[i][0].set_ylabel(f"{noise_types[i]}", rotation="horizontal", ha="right")
            axs[i][int(k * 10)].set_xticks([])
            axs[i][int(k * 10)].set_yticks([])
            axs[i][int(k * 10)].imshow(img)

    plt.show()
