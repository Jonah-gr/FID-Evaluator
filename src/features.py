import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import models, transforms
from torchvision.models.inception import Inception_V3_Weights
from PIL import Image
import pickle


def add_noise(image, noise_level, noise_type):
    if noise_level == 0.0:
        return image
    noisy_image = image + noise_level * torch.randn_like(image)
    noisy_image = torch.clip(noisy_image, 0.0, 1.0)
    return noisy_image


def load_and_preprocess_image(img_path, transform):
    try:
        img = Image.open(img_path).convert("RGB")
        img = transform(img)
        img = img.unsqueeze(0)
        return img
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None


def inception(path, device, noise_levels, noise_types):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.inception_v3(weights=Inception_V3_Weights.DEFAULT)
    model.fc = nn.Identity()
    model.eval()
    model.to(device)

    preprocess = transforms.Compose(
        [
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    features_dict = {k: {} for k in noise_types}
    with torch.no_grad():
        for root, dir, files in os.walk(path):
            for noise_type in noise_types:
                for noise_level in noise_levels:
                    features = []
                    for img_path in tqdm(files, desc=f"Noise Type: {noise_type} | Noise Level: {noise_level}"):
                        img = load_and_preprocess_image(root + "/" + img_path, preprocess)
                        img = img.to(device)
                        img = add_noise(img, noise_level, noise_type)
                        feature = model(img).cpu().numpy().flatten()
                        features.append(feature)
                    features_dict[noise_type][noise_level] = features
    if "no noise" in noise_types:
        return features_dict["no noise"][0.0]
    return features_dict


def compute_features(args):
    real_features = {}
    fake_features = {}

    args.noise = args.noise.split()
    args.noise = [float(num) for num in args.noise]
    if 0.0 not in args.noise:
        args.noise.append(0.0)
    args.noise.sort()

    args.noise_types = args.noise_types.split()
    if "all" in args.noise_types:
        args.noise_types = ["gauss", "blur", "swirl", "rectangles", "salt_and_pepper"]
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
        pickle.dump({"real": {"no pca": real_features}, "fake": {f"no pca": fake_features}}, f)
    print("Features saved to features.pkl")
