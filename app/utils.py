import io
from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_preprocess_transform(image_size: int = 300) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def preprocess_pil_image(image: Image.Image, image_size: int = 300) -> torch.Tensor:
    image = image.convert("RGB")
    transform = get_preprocess_transform(image_size=image_size)
    tensor = transform(image).unsqueeze(0)
    return tensor


def pil_to_numpy_rgb(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))


def normalize_cam(cam: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    cam = cam - cam.min()
    cam = cam / (cam.max() + eps)
    return cam


def generate_heatmap(cam: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    width, height = target_size
    cam_resized = cv2.resize(cam, (width, height))
    cam_uint8 = np.uint8(255 * normalize_cam(cam_resized))
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return heatmap_rgb


def overlay_heatmap_on_image(
    base_image_rgb: np.ndarray, heatmap_rgb: np.ndarray, alpha: float = 0.45
) -> np.ndarray:
    if base_image_rgb.dtype != np.uint8:
        base_image_rgb = np.uint8(np.clip(base_image_rgb, 0, 255))
    overlay = cv2.addWeighted(base_image_rgb, 1 - alpha, heatmap_rgb, alpha, 0)
    return overlay


def np_to_pil(image_array: np.ndarray) -> Image.Image:
    if image_array.dtype != np.uint8:
        image_array = np.uint8(np.clip(image_array, 0, 255))
    return Image.fromarray(image_array)


def image_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()
