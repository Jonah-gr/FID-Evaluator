import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import models, transforms
from torchvision.models.inception import Inception_V3_Weights
from PIL import Image, ImageFilter, ImageDraw
import pickle


def add_noise(image, noise_level, noise_type='noise'):
    if noise_level == 0.0:
        return image
    if noise_type == 'noise':
        noisy_image = image + noise_level * torch.randn_like(image)
        noisy_image = torch.clip(noisy_image, 0.0, 1.0)
    elif noise_type == 'blur':
        noisy_image = image + noise_level * torch.randn_like(image)
        noisy_image = apply_gaussian_blur(image, noise_level)
    elif noise_type == 'rectangles':
        noisy_image = image + noise_level * torch.randn_like(image)
        noisy_image = apply_black_rectangles(image, noise_level)
    
    return noisy_image


def apply_gaussian_blur(image, noise_level):
    pil_image = transforms.ToPILImage()(image).convert("RGB")
    blur_radius_mapped = noise_level * 10
    blurred_image = pil_image.filter(ImageFilter.GaussianBlur(radius=blur_radius_mapped))
    return transforms.ToTensor()(blurred_image)

def apply_black_rectangles(image, noise_level, grid_size=8):
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

def load_and_preprocess_image(img_path, transform):
    try:
        img = Image.open(img_path).convert("RGB")
        img = transform(img)
        img = img.unsqueeze(0)
        return img
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None


def inception(path, device, noise_levels):
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
    features_dict = {}
    with torch.no_grad():
        for root, dir, files in os.walk(path):
            for noise_level in noise_levels:
                features = []
                for img_path in tqdm(files, desc=f"noise_level: {noise_level}"):
                    img = load_and_preprocess_image(root + "/" + img_path, preprocess)
                    img = img.to(device)
                    img = add_noise(img, noise_level)
                    feature = model(img).cpu().numpy().flatten()
                    features.append(feature)
                features_dict[noise_level] = features
    return features_dict


def compute_features(args):
    real_features = {}
    fake_features = {}

    args.noise = args.noise.split()
    args.noise = [float(num) for num in args.noise]
    if args.real:
        print("Real images:")
        real_features = inception(args.real, args.device, args.noise)
        if not args.fake:
            try:
                with open("features.pkl", "rb") as f:
                    features = pickle.load(f)
                    fake_features = features["fake"]["no pca"]
            except:
                print("No previous fake features found")
    if args.fake:
        print("Fake images:")
        fake_features = inception(args.fake, args.device, args.noise)
        if not args.real:
            try:
                with open("features.pkl", "rb") as f:
                    features = pickle.load(f)
                    real_features = features["real"]["no pca"]
            except:
                print("No previous real features found")
    with open("features.pkl", "wb") as f:
        pickle.dump({"real": {"no pca": real_features}, "fake": {f"no pca": fake_features}}, f)
    print("Features saved to features.pkl")
