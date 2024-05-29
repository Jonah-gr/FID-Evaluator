import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import models, transforms
from PIL import Image
import pickle

def load_and_preprocess_image(img_path, transform):
    try:
        img = Image.open(img_path).convert('RGB')
        img = transform(img)
        img = img.unsqueeze(0)
        return img
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None

def compute_features(args):
    if args.device is None:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = models.inception_v3(pretrained=True)
    model.fc = nn.Identity()
    model.eval()
    model.to(args.device)

    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    features = []
    with torch.no_grad():
        for root, dir, files in os.walk(args.path):
            for img_path in tqdm(files):
                img = load_and_preprocess_image(root+"/"+img_path, preprocess)
                img = img.to(args.device)
                feature = model(img).cpu().numpy().flatten()
                features.append(feature)

    with open("features.pkl", 'wb') as f:
        pickle.dump(features, f)
    print("Features saved to features.pkl")
