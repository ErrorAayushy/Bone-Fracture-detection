import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b3

from .cam_methods import generate_all_cams
from .gradcam import GradCAM
from .utils import (
    generate_heatmap,
    np_to_pil,
    overlay_heatmap_on_image,
    pil_to_numpy_rgb,
    preprocess_pil_image,
)


CLASS_NAMES = ["fractured", "not fractured"]


@dataclass
class PredictionOutput:
    label: str
    confidence: float
    confidence_map: Dict[str, float]
    heatmap_image: "Image"
    overlay_image: "Image"
    prediction_time_ms: float


class BoneFracturePredictor:
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        image_size: int = 300,
        class_names: Optional[list] = None,
    ):
        self.image_size = image_size
        self.class_names = class_names or CLASS_NAMES
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path or self._auto_find_checkpoint()
        self.model = self._load_model(self.checkpoint_path).to(self.device).eval()
        self.target_layer = self.model.features[-1]

    @staticmethod
    def _auto_find_checkpoint() -> str:
        project_root = Path(__file__).resolve().parents[1]
        candidates = [
            project_root / "models" / "best_model.pth",
            project_root / "best_model.pth",
            project_root / "model.pth",
            project_root / "weights.pth",
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        discovered = list(project_root.glob("*.pth"))
        if not discovered:
            raise FileNotFoundError(
                "No .pth file found. Place best_model.pth (or model.pth) in project root."
            )
        return str(discovered[0])

    @staticmethod
    def _extract_state_dict(raw_checkpoint) -> Dict[str, torch.Tensor]:
        if isinstance(raw_checkpoint, dict):
            for key in ["state_dict", "model_state_dict", "model"]:
                if key in raw_checkpoint and isinstance(raw_checkpoint[key], dict):
                    raw_checkpoint = raw_checkpoint[key]
                    break
        if not isinstance(raw_checkpoint, dict):
            raise ValueError("Checkpoint format not recognized as a valid state dict.")

        clean = {}
        for k, v in raw_checkpoint.items():
            if not isinstance(v, torch.Tensor):
                continue
            new_k = k[7:] if k.startswith("module.") else k
            clean[new_k] = v
        return clean

    def _build_model(self) -> nn.Module:
        model = efficientnet_b3(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(model.classifier[1].in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 2),
        )
        return model

    def _load_model(self, checkpoint_path: str) -> nn.Module:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        raw_ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = self._extract_state_dict(raw_ckpt)
        model = self._build_model()
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        if missing:
            missing_classifier = [k for k in missing if "classifier" in k]
            if missing_classifier:
                raise RuntimeError(
                    "Classifier weights mismatch. Ensure checkpoint matches EfficientNetB3 "
                    "with 2-class classifier."
                )
        if unexpected:
            unexpected_non_aux = [k for k in unexpected if "num_batches_tracked" not in k]
            if unexpected_non_aux:
                raise RuntimeError(
                    f"Unexpected checkpoint keys detected. Example: {unexpected_non_aux[:3]}"
                )
        return model

    @torch.inference_mode(False)
    def predict_with_gradcam(self, pil_image):
        import time

        start = time.perf_counter()

        image_np = pil_to_numpy_rgb(pil_image)
        h, w = image_np.shape[:2]
        input_tensor = preprocess_pil_image(pil_image, image_size=self.image_size).to(self.device)

        gradcam = GradCAM(self.model, self.target_layer)
        cam, logits = gradcam.generate(input_tensor=input_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)
        pred_idx = int(torch.argmax(probs).item())
        pred_conf = float(probs[pred_idx].item())
        confidence_map = {
            self.class_names[idx]: float(prob.item()) for idx, prob in enumerate(probs)
        }
        gradcam.close()

        heatmap = generate_heatmap(cam, target_size=(w, h))
        overlay = overlay_heatmap_on_image(image_np, heatmap, alpha=0.45)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return PredictionOutput(
            label=self.class_names[pred_idx],
            confidence=pred_conf,
            confidence_map=confidence_map,
            heatmap_image=np_to_pil(heatmap),
            overlay_image=np_to_pil(overlay),
            prediction_time_ms=elapsed_ms,
        )

    @torch.inference_mode(False)
    def predict_with_all_cams(self, pil_image):
        import time

        start = time.perf_counter()
        cam_bundle = generate_all_cams(
            model=self.model,
            image=pil_image,
            target_layer=self.target_layer,
            image_size=self.image_size,
            device=self.device,
        )

        pred_idx = int(cam_bundle["predicted_index"])
        confidence = float(cam_bundle["confidence"])
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        return {
            "label": self.class_names[pred_idx],
            "confidence": confidence,
            "prediction_time_ms": elapsed_ms,
            "cams": cam_bundle["cams"],
        }
