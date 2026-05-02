from pathlib import Path
from typing import Any, List, Optional

import gradio as gr
from PIL import Image

from .model import BoneFracturePredictor
from .reporting import generate_pdf


APP_TITLE = "Bone Fracture Detection AI"
APP_DESC = "EfficientNetB3 + DenseNet121 ensemble with Grad-CAM, Grad-CAM++, and Score-CAM explainability."
DISCLAIMER = "For educational use only. This tool is not a substitute for professional medical diagnosis."
EXPLANATION_MD = """
### Model Explanation
- **What the models detect:** The system uses an ensemble of **EfficientNetB3** and **DenseNet121** to detect fracture-related patterns in X-rays, including cortical breaks and abnormal bone alignment.
- **How prediction is made:** Both models produce class probabilities, and the final decision is made by averaging their probabilities for a more stable result.
- **What CAM visualizations show:** Grad-CAM, Grad-CAM++, and Score-CAM highlight regions that most influenced the prediction, so you can see where the system focused.
- **Why predictions are more reliable:** Combining two architectures with probability averaging and visual explanations improves robustness and transparency.
"""


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
  font-size: 42px;
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
  font-size: 14px;
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
    badge_text = "FRACTURE DETECTED" if is_fractured else "NORMAL"
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
    bar = ("&#9608;" * filled) + ("&#9617;" * (10 - filled))
    return (
        "<div>"
        f"<div style='margin-bottom:6px;font-size:14px;'>Confidence: {bar} {pct:.2f}%</div>"
        f"<div class='confidence-shell'><div class='confidence-fill' style='width:{pct:.2f}%;'></div></div>"
        "</div>"
    )


def run_single_inference(image: Image.Image):
    if image is None:
        raise gr.Error("Please upload an X-ray image first.")

    output = predictor.predict_with_all_cams(image)
    result_html = _result_html(output["label"], output["confidence"], output["prediction_time_ms"])
    confidence_html = _confidence_html(output["confidence"])
    cams = output["cams"]
    return (
        result_html,
        confidence_html,
        image,
        cams["gradcam"]["heatmap"],
        cams["gradcam"]["overlay"],
        cams["gradcam++"]["heatmap"],
        cams["gradcam++"]["overlay"],
        cams["scorecam"]["heatmap"],
        cams["scorecam"]["overlay"],
        {"prediction": output["label"], "confidence": output["confidence"]},
        {
            "gradcam": cams["gradcam"]["heatmap"],
            "gradcam_overlay": cams["gradcam"]["overlay"],
            "gradcam_pp": cams["gradcam++"]["heatmap"],
            "gradcam_pp_overlay": cams["gradcam++"]["overlay"],
            "scorecam": cams["scorecam"]["heatmap"],
            "scorecam_overlay": cams["scorecam"]["overlay"],
        },
    )


def download_report(image: Image.Image, pred_state: dict, cams_state: dict):
    if image is None or not cams_state:
        raise gr.Error("Run a prediction first to generate the report.")
    if not pred_state:
        raise gr.Error("Prediction details are missing. Please run analysis again.")
    prediction = pred_state.get("prediction", "Unknown")
    confidence = float(pred_state.get("confidence", 0.0))
    return generate_pdf(
        image=image,
        prediction=prediction,
        confidence=confidence,
        cams_dict=cams_state,
    )


def _to_path(file_obj: Any) -> str:
    if isinstance(file_obj, str):
        return file_obj
    if hasattr(file_obj, "name"):
        return file_obj.name
    if isinstance(file_obj, dict) and "name" in file_obj:
        return file_obj["name"]
    raise ValueError("Unsupported uploaded file object format.")


def run_batch_inference(files: Optional[List[Any]]):
    if not files:
        raise gr.Error("Please upload one or more X-ray images.")

    rows = []
    gallery = []
    for idx, file_obj in enumerate(files, start=1):
        path = _to_path(file_obj)
        with Image.open(path) as im:
            image = im.convert("RGB")

        output = predictor.predict_with_gradcam(image)
        rows.append(
            [
                idx,
                Path(path).name,
                output.label,
                round(output.confidence * 100.0, 2),
                round(output.prediction_time_ms, 1),
            ]
        )
        caption = f"{Path(path).name} | {output.label} | {output.confidence * 100.0:.2f}%"
        gallery.append((output.heatmap_image, caption))

    return rows, gallery


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

        <span class="meta-chip">Device: {'GPU (CUDA)' if predictor.device.type == 'cuda' else 'CPU'}</span>
        """,
        elem_classes="app-card",
    )

    with gr.Tabs():
        with gr.Tab("Single Prediction"):
            image_input = gr.Image(
                label="Upload Image (X-ray)",
                type="pil",
                sources=["upload"],
                image_mode="RGB",
            )
            predict_btn = gr.Button("Analyze Fracture", variant="primary")
            processing_text = gr.Markdown("**Analyzing X-ray...**", visible=False)
            pred_state = gr.State(value={})
            cams_state = gr.State(value={})

            result_box = gr.HTML(label="Prediction Result")
            confidence_bar = gr.HTML(label="Confidence")
            download_btn = gr.Button("Download Report", variant="secondary")
            report_file = gr.File(label="Downloadable PDF Report")
            gr.Markdown(f"`{DISCLAIMER}`")
            with gr.Accordion("How This Model Works", open=False):
                gr.Markdown(EXPLANATION_MD)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("<div class='label-strong'>ORIGINAL X-RAY</div>")
                    original_out = gr.Image(show_label=False, type="pil")
                with gr.Column():
                    gr.Markdown("<div class='label-strong'>GRAD-CAM HEATMAP</div>")
                    gradcam_heatmap_out = gr.Image(show_label=False, type="pil")
                with gr.Column():
                    gr.Markdown("<div class='label-strong'>GRAD-CAM OVERLAY</div>")
                    gradcam_overlay_out = gr.Image(show_label=False, type="pil")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("<div class='label-strong'>GRAD-CAM++ HEATMAP</div>")
                    gradcampp_heatmap_out = gr.Image(show_label=False, type="pil")
                with gr.Column():
                    gr.Markdown("<div class='label-strong'>GRAD-CAM++ OVERLAY</div>")
                    gradcampp_overlay_out = gr.Image(show_label=False, type="pil")
                with gr.Column():
                    gr.Markdown("<div class='label-strong'>SCORE-CAM HEATMAP</div>")
                    scorecam_heatmap_out = gr.Image(show_label=False, type="pil")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("<div class='label-strong'>SCORE-CAM OVERLAY</div>")
                    scorecam_overlay_out = gr.Image(show_label=False, type="pil")

            examples = collect_examples()
            if examples:
                gr.Examples(
                    examples=examples,
                    inputs=[image_input],
                    outputs=[
                        result_box,
                        confidence_bar,
                        original_out,
                        gradcam_heatmap_out,
                        gradcam_overlay_out,
                        gradcampp_heatmap_out,
                        gradcampp_overlay_out,
                        scorecam_heatmap_out,
                        scorecam_overlay_out,
                        pred_state,
                        cams_state,
                    ],
                    fn=run_single_inference,
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
                    fn=run_single_inference,
                    inputs=[image_input],
                    outputs=[
                        result_box,
                        confidence_bar,
                        original_out,
                        gradcam_heatmap_out,
                        gradcam_overlay_out,
                        gradcampp_heatmap_out,
                        gradcampp_overlay_out,
                        scorecam_heatmap_out,
                        scorecam_overlay_out,
                        pred_state,
                        cams_state,
                    ],
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
            download_btn.click(
                fn=download_report,
                inputs=[original_out, pred_state, cams_state],
                outputs=[report_file],
                api_name="download_report",
                show_progress="full",
            )

        with gr.Tab("Batch Prediction"):
            batch_input = gr.Files(
                label="Upload Multiple X-rays",
                file_count="multiple",
                file_types=["image"],
            )
            batch_btn = gr.Button("Run Batch Prediction", variant="primary")
            batch_processing = gr.Markdown("**Running batch inference...**", visible=False)
            batch_table = gr.Dataframe(
                headers=["#", "Image", "Prediction", "Confidence (%)", "Time (ms)"],
                datatype=["number", "str", "str", "number", "number"],
                label="Batch Results",
            )
            batch_gallery = gr.Gallery(label="Heatmaps (Grad-CAM)", columns=4, height="auto")

            (
                batch_btn.click(
                    fn=lambda: gr.update(visible=True),
                    inputs=None,
                    outputs=[batch_processing],
                    queue=False,
                )
                .then(
                    fn=run_batch_inference,
                    inputs=[batch_input],
                    outputs=[batch_table, batch_gallery],
                    api_name="batch_predict",
                    show_progress="full",
                )
                .then(
                    fn=lambda: gr.update(visible=False),
                    inputs=None,
                    outputs=[batch_processing],
                    queue=False,
                )
            )

if __name__ == "__main__":
    demo.queue(max_size=12).launch()
