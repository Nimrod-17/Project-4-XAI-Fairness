"""Extract image embeddings with a frozen, ImageNet-pretrained backbone.

The CBM heads are trained on top of these embeddings, which makes every training run
take seconds to minutes. Embeddings are computed once and cached as .npz files.

Usage:
    python extract_features.py --data-dir ../release_v0 --image-type derm --backbone resnet50
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm

from data import load_derm7pt

BORDER_CROP = 25  # pixels; some derm7pt images have a black border (same as the official repo)


def build_backbone(name):
    """Return (model without classification head, preprocessing transform, embedding dim)."""
    if name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        dim = model.fc.in_features
        model.fc = torch.nn.Identity()
    elif name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
        dim = model.classifier[1].in_features
        model.classifier = torch.nn.Identity()
    elif name == "vit_b_16":
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        model = models.vit_b_16(weights=weights)
        dim = model.heads.head.in_features
        model.heads = torch.nn.Identity()
    elif name.startswith("dinov2"):
        # Self-supervised ViT (Oquab et al. 2023); forward() returns the CLS embedding.
        model = torch.hub.load("facebookresearch/dinov2", name)
        dim = model.embed_dim
    else:
        raise ValueError(f"Unknown backbone: {name}")

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model.eval(), preprocess, dim


def load_image(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    return img.crop((BORDER_CROP, BORDER_CROP, w - BORDER_CROP, h - BORDER_CROP))


@torch.no_grad()
def extract(paths, model, preprocess, device, batch_size=32):
    feats = []
    for start in tqdm(range(0, len(paths), batch_size), desc="Extracting"):
        batch = torch.stack([preprocess(load_image(p)) for p in paths[start:start + batch_size]])
        feats.append(model(batch.to(device)).cpu().numpy())
    return np.concatenate(feats).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Path to the unzipped release_v0 folder")
    parser.add_argument("--image-type", choices=["derm", "clinic"], default="derm")
    parser.add_argument("--backbone", choices=["resnet50", "efficientnet_b0", "vit_b_16", "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"], default="resnet50")
    parser.add_argument("--out-dir", default="features")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = load_derm7pt(args.data_dir)
    print(dataset.summary())

    model, preprocess, dim = build_backbone(args.backbone)
    model.to(device)
    paths = dataset.image_paths(args.image_type)
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images not found, e.g. {missing[0]}")

    X = extract(paths, model, preprocess, device, args.batch_size)
    assert X.shape == (len(paths), dim)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_file = out / f"{args.backbone}_{args.image_type}.npz"
    # Row i of X corresponds to row i of meta.csv; the row index is stored as a sanity check.
    np.savez_compressed(out_file, X=X, row_index=np.arange(len(paths)))
    print(f"Saved {X.shape} features to {out_file}")


if __name__ == "__main__":
    main()
