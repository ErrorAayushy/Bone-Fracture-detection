from functools import lru_cache
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .densenet_model import DenseNet121Predictor
from .model import BoneFracturePredictor
from .utils import preprocess_pil_image


class EnsemblePredictor:
    def __init__(
        self,
        efficientnet_checkpoint: Optional[str] = None,
        densenet_checkpoint: str = "models/densenet121_v2.pth",
        efficientnet_image_size: int = 300,
        densenet_image_size: int = 224,
    ):
        self.efficientnet_image_size = efficientnet_image_size
        self.densenet_image_size = densenet_image_size
        self.efficientnet = BoneFracturePredictor(
            checkpoint_path=efficientnet_checkpoint, image_size=efficientnet_image_size
        )
        self.densenet = DenseNet121Predictor(
            checkpoint_path=densenet_checkpoint, image_size=densenet_image_size
        )
        self.class_names = self.efficientnet.class_names
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.efficientnet.model = self.efficientnet.model.to(self.device).eval()
        self.densenet.model = self.densenet.model.to(self.device).eval()

    @torch.inference_mode()
    def ensemble_predict(self, image) -> Tuple[str, float, Dict[str, float]]:
        input_tensor_eff = preprocess_pil_image(
            image, image_size=self.efficientnet_image_size
        ).to(self.device)
        input_tensor_den = preprocess_pil_image(
            image, image_size=self.densenet_image_size
        ).to(self.device)

        probs_eff = F.softmax(self.efficientnet.model(input_tensor_eff), dim=1)
        probs_den = F.softmax(self.densenet.model(input_tensor_den), dim=1)
        final_probs = (probs_eff + probs_den) / 2.0

        final_probs = final_probs.squeeze(0)
        pred_idx = int(torch.argmax(final_probs).item())
        confidence = float(final_probs[pred_idx].item())
        confidence_map = {
            self.class_names[i]: float(final_probs[i].item()) for i in range(len(self.class_names))
        }
        return self.class_names[pred_idx], confidence, confidence_map


@lru_cache(maxsize=1)
def get_ensemble_predictor() -> EnsemblePredictor:
    # Cache model loading to avoid repeated startup cost and inference slowdown.
    return EnsemblePredictor()


def ensemble_predict(image) -> Tuple[str, float]:
    predictor = get_ensemble_predictor()
    label, confidence, _ = predictor.ensemble_predict(image)
    return label, confidence
