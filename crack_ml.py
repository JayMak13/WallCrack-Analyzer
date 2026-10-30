"""Local ML utilities for wall-crack classification.

The model is deliberately optional at runtime.  If PyTorch or a trained model
is unavailable, callers can continue using the application's rule-based
analyser.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import torch
    from PIL import Image
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "crack_classifier.pt"
CLASS_NAMES = [
    "diagonal",
    "hairline_shrinkage",
    "horizontal",
    "map_crazing",
    "stair_step",
    "vertical_settlement_minor",
]

# Dataset folder names are intentionally separate from user-facing/knowledge-base
# names, so these can evolve independently.
FOLDER_TO_CLASS = {
    "DIAGONAL OR INCLINED": "diagonal",
    "HAIRLINE": "hairline_shrinkage",
    "HORIZONTAL": "horizontal",
    "MAP PATTERN": "map_crazing",
    "STAIR STEP": "stair_step",
    "VERTICAL": "vertical_settlement_minor",
}


def build_model(class_count: int):
    """Return the same MobileNetV3 architecture used by the trainer."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch and torchvision are not installed.")
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, class_count)
    return model


def predict_image(image_path: str | Path, model_path: Path = MODEL_PATH) -> Optional[Dict[str, Any]]:
    """Classify one image; return None when no local model is ready."""
    if not TORCH_AVAILABLE or not model_path.exists():
        return None
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        # Never put an unproven training run in front of the established rules.
        # The trainer records held-out validation accuracy in every checkpoint.
        if checkpoint.get("validation_accuracy", 0.0) < 0.65:
            return None
        class_names = checkpoint.get("class_names", CLASS_NAMES)
        model = build_model(len(class_names))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        transform = transforms.Compose([
            transforms.Resize(160),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        with Image.open(image_path).convert("RGB") as image:
            batch = transform(image).unsqueeze(0)
        with torch.inference_mode():
            probabilities = torch.softmax(model(batch)[0], dim=0).tolist()
        ranked = sorted(enumerate(probabilities), key=lambda item: item[1], reverse=True)
        best_index, confidence = ranked[0]
        return {
            "primary_type": class_names[best_index],
            "primary_confidence": float(confidence),
            "all_scores": {class_names[index]: float(score) for index, score in ranked},
            "reasoning": "Classification from the locally trained image model.",
            "model_version": checkpoint.get("model_version", "unknown"),
        }
    except Exception:
        # Inference must never prevent the established analyser from working.
        return None


def save_high_confidence_candidate(
    image_path: str | Path,
    classification: Dict[str, Any],
    threshold: float = 0.95,
) -> bool:
    """Privately queue a high-confidence prediction for later batch retraining.

    These images are *not* added to the active training set automatically.
    ``train_model.py --include-candidates`` is an explicit, offline promotion
    step and still retains the original labelled dataset.
    """
    if classification.get("primary_confidence", 0) < threshold:
        return False
    label = classification.get("primary_type")
    if label not in CLASS_NAMES:
        return False
    source = Path(image_path)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = PROJECT_DIR / "data" / "candidate_uploads" / label / f"{digest}.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return False
    try:
        with Image.open(source).convert("RGB") as image:
            image.save(destination, "PNG", optimize=True)
        record = {
            "file": str(destination.relative_to(PROJECT_DIR)).replace("\\", "/"),
            "predicted_label": label,
            "confidence": round(float(classification["primary_confidence"]), 6),
            "model_version": classification.get("model_version", "rule-based"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with (destination.parent.parent / "manifest.jsonl").open("a", encoding="utf-8") as manifest:
            manifest.write(json.dumps(record) + "\n")
        return True
    except Exception:
        destination.unlink(missing_ok=True)
        return False
