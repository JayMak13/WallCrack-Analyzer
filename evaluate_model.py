"""Evaluate the trained classifier on the deterministic held-out split.

Run after training: python evaluate_model.py
The output intentionally includes per-class recall, because headline accuracy
can hide a model that ignores a less-common crack category.
"""
from __future__ import annotations

import json

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from crack_ml import CLASS_NAMES, MODEL_PATH, build_model
from train_model import CrackDataset, stratified_split, find_samples


def main():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    class_names = checkpoint["class_names"]
    _, validation_samples = stratified_split(find_samples(False))
    transform = transforms.Compose([
        transforms.Resize(160), transforms.CenterCrop(128), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    loader = DataLoader(CrackDataset(validation_samples, transform), batch_size=32)
    model = build_model(len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    matrix = [[0 for _ in class_names] for _ in class_names]
    with torch.inference_mode():
        for images, labels in loader:
            predicted = model(images).argmax(1)
            for actual, prediction in zip(labels.tolist(), predicted.tolist()):
                matrix[actual][prediction] += 1
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[index][index] for index in range(len(class_names)))
    recalls = {
        class_names[index]: round(matrix[index][index] / max(sum(matrix[index]), 1), 4)
        for index in range(len(class_names))
    }
    report = {
        "model_version": checkpoint.get("model_version"),
        "validation_images": total,
        "accuracy": round(correct / max(total, 1), 4),
        "per_class_recall": recalls,
        "confusion_matrix": {class_names[index]: matrix[index] for index in range(len(class_names))},
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
