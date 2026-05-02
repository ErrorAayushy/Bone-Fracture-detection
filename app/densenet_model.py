from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121

from .utils import preprocess_pil_image


CLASS_NAMES = ["fractured", "not fractured"]


class DenseNet121Predictor:
    def __init__(
        self,
        checkpoint_path: str = "models/densenet121_v2.pth",
        image_size: int = 224,
        class_names: Optional[list] = None,
    ):
        self.image_size = image_size
        self.class_names = class_names or CLASS_NAMES
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = self._resolve_checkpoint(checkpoint_path)
        self.model = self._build_model()
        self._load_weights(self.checkpoint_path)
        self.model = self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _resolve_checkpoint(checkpoint_path: str) -> str:
        root = Path(__file__).resolve().parents[1]
        if checkpoint_path in {"models/densenet121.pth", "models/dense121_v2.pth"}:
            checkpoint_path = "models/densenet121_v2.pth"
        candidate = root / checkpoint_path
        if candidate.exists():
            return str(candidate)
        fallback = Path(checkpoint_path)
        if fallback.exists():
            return str(fallback.resolve())
        raise FileNotFoundError(f"DenseNet checkpoint not found at: {checkpoint_path}")

    @staticmethod
    def _extract_state_dict(raw_checkpoint) -> Dict[str, torch.Tensor]:
        if isinstance(raw_checkpoint, dict):
            for key in ["state_dict", "model_state_dict", "model"]:
                if key in raw_checkpoint and isinstance(raw_checkpoint[key], dict):
                    raw_checkpoint = raw_checkpoint[key]
                    break
        if not isinstance(raw_checkpoint, dict):
            raise ValueError("Checkpoint format not recognized as a valid state_dict.")

        clean_state_dict: Dict[str, torch.Tensor] = {}
        for key, value in raw_checkpoint.items():
            if not isinstance(value, torch.Tensor):
                continue
            cleaned_key = key[7:] if key.startswith("module.") else key
            clean_state_dict[cleaned_key] = value
        return clean_state_dict

    @staticmethod
    def _build_model() -> nn.Module:
        model = densenet121(pretrained=False)
        model.classifier = nn.Linear(1024, 2)
        return model

    def _load_weights(self, checkpoint_path: str) -> None:
        raw_checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = self._extract_state_dict(raw_checkpoint)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)

        missing_classifier = [k for k in missing if "classifier" in k]
        if missing_classifier:
            raise RuntimeError(
                "DenseNet classifier weights mismatch. Expected nn.Linear(1024, 2)."
            )

        unexpected_non_aux = [k for k in unexpected if "num_batches_tracked" not in k]
        if unexpected_non_aux:
            raise RuntimeError(
                f"Unexpected DenseNet checkpoint keys. Example: {unexpected_non_aux[:3]}"
            )

    @torch.inference_mode()
    def predict(self, image) -> Tuple[str, float]:
        input_tensor = preprocess_pil_image(image, image_size=self.image_size).to(self.device)
        logits = self.model(input_tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0)
        pred_idx = int(torch.argmax(probabilities).item())
        pred_prob = float(probabilities[pred_idx].item())
        return self.class_names[pred_idx], pred_prob
