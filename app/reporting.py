import tempfile
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def _fit_size(orig_w, orig_h, max_w, max_h):
    if orig_w <= 0 or orig_h <= 0:
        return max_w, max_h
    ratio = min(max_w / float(orig_w), max_h / float(orig_h))
    return orig_w * ratio, orig_h * ratio


def _draw_labeled_image(c, img, label, x, y_top, box_w, box_h):
    c.setFont("Helvetica-Bold", 9)
    c.drawString(x, y_top, label)

    image_top = y_top - 10
    image_bottom = image_top - box_h
    c.rect(x, image_bottom, box_w, box_h)

    if img is not None:
        reader = ImageReader(img)
        iw, ih = img.size
        dw, dh = _fit_size(iw, ih, box_w, box_h)
        c.drawImage(
            reader,
            x + (box_w - dw) / 2.0,
            image_bottom + (box_h - dh) / 2.0,
            width=dw,
            height=dh,
            preserveAspectRatio=True,
            mask="auto",
        )
    return image_bottom


def generate_pdf(image, prediction, confidence, cams_dict):
    if image is None:
        raise ValueError("Original image is required for report generation.")
    required = [
        "gradcam",
        "gradcam_overlay",
        "gradcam_pp",
        "gradcam_pp_overlay",
        "scorecam",
        "scorecam_overlay",
    ]
    missing = [k for k in required if k not in cams_dict or cams_dict[k] is None]
    if missing:
        raise ValueError(f"Missing CAM images for report: {missing}")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf_path = tmp.name
    tmp.close()

    c = canvas.Canvas(pdf_path, pagesize=A4)
    page_w, page_h = A4
    margin = 36

    y = page_h - margin
    c.setFont("Helvetica-Bold", 18)
    c.drawString(margin, y, "Bone Fracture Detection Report")

    y -= 20
    c.setFont("Helvetica", 10)
    c.drawString(margin, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    y -= 24
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Prediction Result")
    y -= 16
    c.setFont("Helvetica", 11)
    c.drawString(margin, y, f"Predicted class: {prediction}")
    y -= 16
    c.drawString(margin, y, f"Confidence: {confidence * 100:.2f}%")

    y -= 26
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, y, "Visual Explanations")

    grid_items = [
        ("Original X-ray", image),
        ("Grad-CAM Heatmap", cams_dict["gradcam"]),
        ("Grad-CAM Overlay", cams_dict["gradcam_overlay"]),
        ("Grad-CAM++ Heatmap", cams_dict["gradcam_pp"]),
        ("Grad-CAM++ Overlay", cams_dict["gradcam_pp_overlay"]),
        ("Score-CAM Heatmap", cams_dict["scorecam"]),
        ("Score-CAM Overlay", cams_dict["scorecam_overlay"]),
    ]

    y -= 16
    row_gap = 12
    label_gap = 10
    col_gap = 12
    box_w = (page_w - (2 * margin) - col_gap) / 2.0
    box_h = 108

    idx = 0
    while idx < len(grid_items):
        left_label, left_img = grid_items[idx]
        right_item = grid_items[idx + 1] if idx + 1 < len(grid_items) else None
        right_label, right_img = right_item if right_item else ("", None)

        needed_h = label_gap + box_h + row_gap
        if y - needed_h < margin + 30:
            c.showPage()
            y = page_h - margin
            c.setFont("Helvetica-Bold", 11)
            c.drawString(margin, y, "Visual Explanations (continued)")
            y -= 16

        left_bottom = _draw_labeled_image(c, left_img, left_label, margin, y, box_w, box_h)
        if right_item:
            _draw_labeled_image(c, right_img, right_label, margin + box_w + col_gap, y, box_w, box_h)
        y = left_bottom - row_gap
        idx += 2

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(
        margin,
        margin,
        "Educational use only. Not a substitute for professional medical diagnosis.",
    )

    c.showPage()
    c.save()
    return pdf_path
