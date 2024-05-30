import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import models, transforms
from PIL import Image
import pickle


def add_noise(images, noise_level):
    noisy_images = images + noise_level * torch.randn_like(images)
    noisy_images = torch.clip(noisy_images, 0., 1.)
    return noisy_images

def load_and_preprocess_image(img_path, transform):
    try:
        img = Image.open(img_path).convert('RGB')
        img = transform(img)
        img = img.unsqueeze(0)
        return img
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None
    
def inception(path, device, noise_levels, real=True):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = models.inception_v3(pretrained=True)
    model.fc = nn.Identity()
    model.eval()
    model.to(device)

    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    features_dict = {"real": {}} if real else {"fake": {}}
    with torch.no_grad():
        for root, dir, files in os.walk(path):
            for noise_level in noise_levels:
                features = []
                for img_path in tqdm(files, desc=f"noise_level: {noise_level}"):
                    img = load_and_preprocess_image(root+"/"+img_path, preprocess)
                    img = img.to(device)
                    img = add_noise(img, noise_level)
                    feature = model(img).cpu().numpy().flatten()
                    features.append(feature) 
                features_dict[list(features_dict)[0]][noise_level] = features
    return features_dict

def compute_features(args, save=True):
    features = inception(args.path, args.device, args.noise)
    if save:
        with open("features.pkl", 'wb') as f:
            pickle.dump(features, f)
        print("Features saved to features.pkl")
