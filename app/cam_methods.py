from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .utils import (
    generate_heatmap,
    normalize_cam,
    np_to_pil,
    overlay_heatmap_on_image,
    pil_to_numpy_rgb,
    preprocess_pil_image,
)


class _ActivationGradientHook:
    def __init__(self, target_layer: torch.nn.Module):
        self.activations = None
        self.gradients = None
        self._fh = target_layer.register_forward_hook(self._forward_hook)
        self._bh = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, _module, _inputs, outputs):
        self.activations = outputs

    def _backward_hook(self, _module, _grad_inputs, grad_outputs):
        self.gradients = grad_outputs[0]

    def close(self):
        self._fh.remove()
        self._bh.remove()


def _resolve_target_layer(model: torch.nn.Module, target_layer: Optional[torch.nn.Module]):
    if target_layer is not None:
        return target_layer
    if hasattr(model, "features") and len(model.features) > 0:
        return model.features[-1]
    raise ValueError("Could not infer target layer. Pass target_layer explicitly.")


def _gradcam_from_tensors(activations: torch.Tensor, gradients: torch.Tensor):
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * activations, dim=1, keepdim=True)
    cam = F.relu(cam)
    return cam.squeeze(0).squeeze(0).detach().cpu().numpy()


def _gradcam_pp_from_tensors(
    activations: torch.Tensor, gradients: torch.Tensor, eps: float = 1e-8
):
    grad2 = gradients.pow(2)
    grad3 = gradients.pow(3)
    sum_acts = torch.sum(activations, dim=(2, 3), keepdim=True)
    denom = 2.0 * grad2 + sum_acts * grad3 + eps
    alpha = grad2 / denom
    weights = torch.sum(alpha * F.relu(gradients), dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * activations, dim=1, keepdim=True)
    cam = F.relu(cam)
    return cam.squeeze(0).squeeze(0).detach().cpu().numpy()


@torch.inference_mode()
def _scorecam_from_activations(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    activations: torch.Tensor,
    class_idx: int,
    max_channels: int = 32,
):
    acts = activations.detach()
    acts = F.relu(acts)
    num_channels = acts.shape[1]
    if num_channels == 0:
        raise RuntimeError("Score-CAM failed: no channels found in target layer activations.")

    if num_channels > max_channels:
        channel_scores = acts.mean(dim=(2, 3)).squeeze(0)
        topk_idx = torch.topk(channel_scores, k=max_channels).indices
        acts = acts[:, topk_idx, :, :]

    upsampled = F.interpolate(
        acts, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False
    )

    n = upsampled.shape[1]
    flat = upsampled.view(n, -1)
    mins = flat.min(dim=1, keepdim=True).values.view(1, n, 1, 1)
    maxs = flat.max(dim=1, keepdim=True).values.view(1, n, 1, 1)
    masks = (upsampled - mins) / (maxs - mins + 1e-8)
    masks = masks.clamp(0.0, 1.0)

    masked_inputs = input_tensor.repeat(n, 1, 1, 1) * masks.permute(1, 0, 2, 3)
    logits = model(masked_inputs)
    scores = F.softmax(logits, dim=1)[:, class_idx]
    weights = scores.view(1, n, 1, 1)

    cam = torch.sum(weights * acts, dim=1, keepdim=True)
    cam = F.relu(cam)
    return cam.squeeze(0).squeeze(0).detach().cpu().numpy()


def generate_all_cams(
    model: torch.nn.Module,
    image,
    target_layer: Optional[torch.nn.Module] = None,
    image_size: int = 300,
    device: Optional[torch.device] = None,
    class_idx: Optional[int] = None,
    scorecam_max_channels: int = 32,
    overlay_alpha: float = 0.45,
) -> Dict:
    """
    Generate Grad-CAM, Grad-CAM++, and Score-CAM heatmaps/overlays.
    Returns a dictionary with PIL outputs for easy UI usage.
    """
    if image is None:
        raise ValueError("Input image is required.")

    target_layer = _resolve_target_layer(model, target_layer)
    model_device = next(model.parameters()).device
    device = device or model_device

    image_np = pil_to_numpy_rgb(image)
    h, w = image_np.shape[:2]
    input_tensor = preprocess_pil_image(image, image_size=image_size).to(device)

    model.zero_grad(set_to_none=True)
    hook = _ActivationGradientHook(target_layer)
    try:
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)
        pred_idx = int(torch.argmax(probs).item()) if class_idx is None else int(class_idx)
        confidence = float(probs[pred_idx].item())

        score = logits[:, pred_idx]
        score.backward(retain_graph=True)
        if hook.activations is None or hook.gradients is None:
            raise RuntimeError("CAM hooks failed to capture activations/gradients.")

        gradcam_raw = _gradcam_from_tensors(hook.activations, hook.gradients)
        gradcam_pp_raw = _gradcam_pp_from_tensors(hook.activations, hook.gradients)

        scorecam_raw = _scorecam_from_activations(
            model=model,
            input_tensor=input_tensor,
            activations=hook.activations,
            class_idx=pred_idx,
            max_channels=scorecam_max_channels,
        )
    finally:
        hook.close()
        model.zero_grad(set_to_none=True)

    cams = {}
    for key, raw_cam in {
        "gradcam": gradcam_raw,
        "gradcam++": gradcam_pp_raw,
        "scorecam": scorecam_raw,
    }.items():
        normalized = normalize_cam(raw_cam)
        heatmap = generate_heatmap(normalized, target_size=(w, h))
        overlay = overlay_heatmap_on_image(image_np, heatmap, alpha=overlay_alpha)
        cams[key] = {
            "heatmap": np_to_pil(heatmap),
            "overlay": np_to_pil(overlay),
        }

    return {
        "predicted_index": pred_idx,
        "confidence": confidence,
        "cams": cams,
    }
