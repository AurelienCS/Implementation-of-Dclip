import os
import sys
import subprocess
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

TEMP_BASE = "/tempory/cg/deepl/dclip"
custom_package_path = os.path.join(TEMP_BASE, "package")
os.makedirs(custom_package_path, exist_ok=True)

if custom_package_path not in sys.path:
    sys.path.append(custom_package_path)

def install_if_missing(package, pip_name=None):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--target=" + custom_package_path, pip_name or package])

install_if_missing("clip", "git+https://github.com/openai/CLIP.git")
install_if_missing("datasets")

import clip
from datasets import load_dataset
from dclip import dclip
from train import train_dclip
from utils import patch_vit_resolution, visualize_comparison_clip_dclip, VOCHFDataset

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = "ViT-B/32"
    class_names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    clip_std, clip_preprocess = clip.load(backbone, device=device)
    clip_std = patch_vit_resolution(clip_std, new_resolution=448)
    
    model = dclip(classnames=class_names, clip_type=backbone).to(device)

    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275)),
    ])

    # Utilisation du chemin mteb/voc2007 qui est très stable sur le hub
    dataset = load_dataset("mteb/voc2007", cache_dir=os.path.join(TEMP_BASE, "hf_datasets"))
    
    train_loader = DataLoader(VOCHFDataset(dataset["train"], class_names, transform=transform), batch_size=32, shuffle=True)
    test_loader = DataLoader(VOCHFDataset(dataset["test"], class_names, transform=transform), batch_size=32)

    # Extraction d'une image de test pour la visualisation
    sample = dataset["test"][0]
    test_img = sample["image"]
    test_labels = [class_names[i] for i in sample["label"]] if "label" in sample else []

    n = 6
    for epoch in range(1, n):
        train_dclip(model, train_loader, test_loader, class_names, num_epochs=1, device=device, alpha=7e-5)
        visualize_comparison_clip_dclip(clip_std, clip_preprocess, model, test_img, test_labels, device=device)
