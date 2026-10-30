"""Train the local wall-crack classifier from Sample_Images.

Usage:
    python train_model.py
    python train_model.py --epochs 12 --include-candidates
"""
from __future__ import annotations

import argparse
import os
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Keep pretrained-weight downloads within the project instead of a user-profile
# cache, which also makes a deployment build more predictable.
os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parent / ".torch"))

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from crack_ml import CLASS_NAMES, FOLDER_TO_CLASS, MODEL_PATH, PROJECT_DIR

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class CrackDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples, self.transform = samples, transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        with Image.open(path).convert("RGB") as image:
            return self.transform(image), target


def find_samples(include_candidates: bool):
    samples = []
    root = PROJECT_DIR / "Sample_Images"
    for folder, label in FOLDER_TO_CLASS.items():
        for path in (root / folder).rglob("*"):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((path, CLASS_NAMES.index(label)))
    if include_candidates:
        candidate_root = PROJECT_DIR / "data" / "candidate_uploads"
        for label in CLASS_NAMES:
            for path in (candidate_root / label).glob("*") if (candidate_root / label).exists() else []:
                if path.suffix.lower() in IMAGE_EXTENSIONS:
                    samples.append((path, CLASS_NAMES.index(label)))
    return samples


def stratified_split(samples, validation_ratio=0.2, seed=42):
    by_label = {index: [] for index in range(len(CLASS_NAMES))}
    for sample in samples:
        by_label[sample[1]].append(sample)
    train, validation = [], []
    rng = random.Random(seed)
    for label_samples in by_label.values():
        rng.shuffle(label_samples)
        split_at = max(1, round(len(label_samples) * validation_ratio))
        validation.extend(label_samples[:split_at])
        train.extend(label_samples[split_at:])
    return train, validation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--include-candidates", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(42)
    samples = find_samples(args.include_candidates)
    if len(samples) < len(CLASS_NAMES) * 2:
        raise SystemExit("Not enough labelled images were found in Sample_Images.")
    train_samples, validation_samples = stratified_split(samples)
    print(f"Images: {len(samples)} | train: {len(train_samples)} | validation: {len(validation_samples)}")
    print("Class counts:", {CLASS_NAMES[key]: value for key, value in Counter(y for _, y in samples).items()})

    train_transform = transforms.Compose([
        transforms.Resize(160), transforms.RandomResizedCrop(128, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=.2, contrast=.2, saturation=.1), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(160), transforms.CenterCrop(128), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_loader = DataLoader(CrackDataset(train_samples, train_transform), batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(CrackDataset(validation_samples, eval_transform), batch_size=args.batch_size)
    try:
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    except Exception as error:
        print(f"Could not fetch pretrained weights ({error}); training without them.")
        model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(CLASS_NAMES))
    # Transfer learning: the pretrained feature extractor already understands
    # visual texture and edges; this small dataset trains only the new head.
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_accuracy, best_state = -1.0, None
    for epoch in range(args.epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        model.eval(); correct = total = 0
        with torch.inference_mode():
            for images, labels in validation_loader:
                predicted = model(images.to(device)).argmax(1).cpu()
                correct += (predicted == labels).sum().item(); total += len(labels)
        accuracy = correct / max(total, 1)
        print(f"Epoch {epoch + 1}/{args.epochs}: validation accuracy {accuracy:.1%}")
        if accuracy > best_accuracy:
            best_accuracy, best_state = accuracy, {key: value.cpu() for key, value in model.state_dict().items()}
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": best_state, "class_names": CLASS_NAMES,
        "validation_accuracy": best_accuracy, "training_images": len(samples),
        "model_version": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    }, MODEL_PATH)
    print(f"Saved {MODEL_PATH} (best validation accuracy: {best_accuracy:.1%})")


if __name__ == "__main__":
    main()
