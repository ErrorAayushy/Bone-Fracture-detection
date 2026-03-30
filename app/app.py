from pathlib import Path

import gradio as gr
from PIL import Image

from .model import BoneFracturePredictor


APP_TITLE = "Bone Fracture Detection AI"
APP_DESC = "EfficientNetB3 + Grad-CAM interpretability for fracture screening on X-rays."
MODEL_ACCURACY_LABEL = "Model Accuracy: ~97%"
DISCLAIMER = "For educational use only. This tool is not a substitute for professional medical diagnosis."


CUSTOM_CSS = """
body, .gradio-container {
  background: radial-gradient(1200px 600px at 15% -20%, #14273f 0%, #0a121b 45%, #070d14 100%);
  color: #e8f0f8;
}
.app-card {
  border: 1px solid rgba(104, 159, 196, 0.25);
  border-radius: 16px;
  background: rgba(11, 21, 32, 0.8);
  backdrop-filter: blur(6px);
}
.meta-chip {
  display: inline-block;
  padding: 6px 12px;
  border-radius: 999px;
  border: 1px solid rgba(90, 167, 193, 0.5);
  margin-right: 8px;
  font-size: 12px;
  color: #bfeaff;
}
.confidence-shell {
  width: 100%;
  background: #122233;
  border: 1px solid #2e4961;
  border-radius: 10px;
  height: 16px;
  overflow: hidden;
}
.confidence-fill {
  height: 100%;
  background: linear-gradient(90deg, #3ca5ff 0%, #18e2b2 100%);
  transition: width 0.25s ease;
}
.result-badge {
  text-align: center;
  padding: 16px 18px;
  border-radius: 12px;
  border: 1px solid rgba(79, 184, 215, 0.5);
  background: rgba(7, 39, 52, 0.55);
}
.result-main {
  font-size: 44px;
  font-weight: 800;
  letter-spacing: 1px;
  line-height: 1.1;
}
.result-sub {
  margin-top: 8px;
  font-size: 14px;
  color: #c7d8e8;
}
.label-strong {
  text-align: center;
  font-size: 15px;
  font-weight: 800;
  letter-spacing: 0.8px;
  margin-bottom: 6px;
  color: #d4e9f7;
}
"""


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "best_model.pth"
predictor = BoneFracturePredictor(
    checkpoint_path=str(CHECKPOINT_PATH) if CHECKPOINT_PATH.exists() else None
)


def _result_html(label: str, confidence: float, time_ms: float) -> str:
    is_fractured = label.lower() == "fractured"
    label_text = "FRACTURED" if is_fractured else "NOT FRACTURED"
    label_color = "#ff5f5f" if is_fractured else "#49d17d"
    badge_text = "&#128308; FRACTURE DETECTED" if is_fractured else "&#128994; NORMAL"
    return (
        "<div class='result-badge'>"
        f"<div class='result-main' style='color:{label_color};'>{label_text}</div>"
        f"<div class='result-sub'>{badge_text}</div>"
        f"<div class='result-sub'>Inference Time: {time_ms:.1f} ms</div>"
        "</div>"
    )


def _confidence_html(confidence: float) -> str:
    pct = max(0.0, min(100.0, confidence * 100.0))
    filled = int(round(pct / 10))
    bar = ("█" * filled) + ("░" * (10 - filled))
    return (
        "<div>"
        f"<div style='margin-bottom:6px;font-size:14px;'>Confidence: {bar} {pct:.2f}%</div>"
        f"<div class='confidence-shell'><div class='confidence-fill' style='width:{pct:.2f}%;'></div></div>"
        "</div>"
    )


def run_inference(image: Image.Image):
    if image is None:
        raise gr.Error("Please upload an X-ray image first.")

    output = predictor.predict_with_gradcam(image)
    result_html = _result_html(output.label, output.confidence, output.prediction_time_ms)
    confidence_html = _confidence_html(output.confidence)
    return result_html, confidence_html, image, output.heatmap_image, output.overlay_image


def collect_examples():
    folder = PROJECT_ROOT / "examples"
    if not folder.exists():
        return None
    image_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        image_paths.extend(folder.glob(ext))
    if not image_paths:
        return None
    return [[str(p)] for p in sorted(image_paths)]


with gr.Blocks(
    title=APP_TITLE,
    theme=gr.themes.Base(primary_hue="cyan", neutral_hue="slate"),
    css=CUSTOM_CSS,
) as demo:
    gr.Markdown(
        f"""
        # {APP_TITLE}
        {APP_DESC}

        <span class="meta-chip">{MODEL_ACCURACY_LABEL}</span>
        <span class="meta-chip">Device: {'GPU (CUDA)' if predictor.device.type == 'cuda' else 'CPU'}</span>
        """,
        elem_classes="app-card",
    )

    image_input = gr.Image(
        label="Upload Image (X-ray)",
        type="pil",
        sources=["upload"],
        image_mode="RGB",
    )
    predict_btn = gr.Button("🔍 Analyze Fracture", variant="primary")
    processing_text = gr.Markdown("**Analyzing X-ray...**", visible=False)

    result_box = gr.HTML(label="Prediction Result")
    confidence_bar = gr.HTML(label="Confidence")
    gr.Markdown(f"`{DISCLAIMER}`")

    with gr.Row():
        with gr.Column():
            gr.Markdown("<div class='label-strong'>ORIGINAL X-RAY</div>")
            original_out = gr.Image(show_label=False, type="pil")
        with gr.Column():
            gr.Markdown("<div class='label-strong'>MODEL ATTENTION</div>")
            heatmap_out = gr.Image(show_label=False, type="pil")
        with gr.Column():
            gr.Markdown("<div class='label-strong'>FRACTURE LOCALIZATION</div>")
            overlay_out = gr.Image(show_label=False, type="pil")

    examples = collect_examples()
    if examples:
        gr.Examples(
            examples=examples,
            inputs=[image_input],
            outputs=[result_box, confidence_bar, original_out, heatmap_out, overlay_out],
            fn=run_inference,
            cache_examples=False,
            label="Example X-ray Images",
        )

    (
        predict_btn.click(
            fn=lambda: gr.update(visible=True),
            inputs=None,
            outputs=[processing_text],
            queue=False,
        )
        .then(
            fn=run_inference,
            inputs=[image_input],
            outputs=[result_box, confidence_bar, original_out, heatmap_out, overlay_out],
            api_name="predict",
            show_progress="full",
        )
        .then(
            fn=lambda: gr.update(visible=False),
            inputs=None,
            outputs=[processing_text],
            queue=False,
        )
    )


if __name__ == "__main__":
    demo.queue(max_size=12).launch(server_name="0.0.0.0", share=True)
