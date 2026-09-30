import base64
import io
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from flask import Flask, render_template, request
from PIL import Image
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

BASE_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = BASE_DIR / "models" / "fusion_bundle.joblib"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"
SAMPLE_DIR = BASE_DIR / "static" / "samples"

LINKEDIN_URL = "https://www.linkedin.com/in/infinitepraveen"
GITHUB_URL = "https://github.com/InfinitePraveen"

ALLOWED_EXT = {"jpg", "jpeg", "png", "webp", "gif"}
MAX_TEXT_CHARS = 500

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 6 * 1024 * 1024  # 6 MB upload limit

state = {"bundle": None, "net": None, "prep": None, "error": None}


def clean_text(t):
    # same function as in notebook 01, keep the two in sync
    t = t.lower()
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"\brt\b", " ", t)
    t = t.replace("#", " ")
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def load_models():
    if not BUNDLE_PATH.exists():
        state["error"] = (
            "No trained model found in models/. Run the three notebooks in order "
            "(01, 02, 03) first, then restart the app."
        )
        return

    state["bundle"] = joblib.load(BUNDLE_PATH)

    weights = MobileNet_V3_Small_Weights.DEFAULT
    net = mobilenet_v3_small(weights=weights)
    net.classifier = torch.nn.Identity()
    net.eval()
    state["net"] = net
    state["prep"] = weights.transforms()


def image_vector(img):
    with torch.no_grad():
        out = state["net"](torch.stack([state["prep"](img)])).numpy()
    return out.reshape(1, -1)


def get_samples():
    csv_path = SAMPLE_DIR / "samples.csv"
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path).fillna("")
    return df.to_dict(orient="records")


def get_metrics():
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return None


def predict(text, img):
    b = state["bundle"]
    classes = b["classes"]
    cleaned = clean_text(text) if text else ""

    p_text = p_img = None
    tf_vec = img_vec = None

    if cleaned:
        tf_vec = b["tfidf"].transform([cleaned])
        if tf_vec.nnz > 0:  # none of the words were in the vocabulary -> no text signal
            p_text = b["text_clf"].predict_proba(tf_vec)[0]
    if img is not None:
        img_vec = image_vector(img)
        p_img = b["image_clf"].predict_proba(b["img_scaler"].transform(img_vec))[0]

    if p_text is not None and p_img is not None:
        mode = "fusion"
        if b["deploy"] == "late":
            p_final = b["late_w"] * p_text + (1 - b["late_w"]) * p_img
        else:
            z = np.hstack([b["svd"].transform(tf_vec), img_vec])
            p_final = b["early_clf"].predict_proba(b["fuse_scaler"].transform(z))[0]
    elif p_text is not None:
        mode, p_final = "text only", p_text
    elif p_img is not None:
        mode, p_final = "image only", p_img
    else:
        return None

    def pack(p):
        if p is None:
            return None
        rows = [{"label": c, "pct": round(float(v) * 100, 1)} for c, v in zip(classes, p)]
        top = max(rows, key=lambda r: r["pct"])
        return {"rows": rows, "top": top["label"], "top_pct": top["pct"]}

    return {"mode": mode, "final": pack(p_final), "text": pack(p_text), "image": pack(p_img)}


def to_data_uri(img):
    buf = io.BytesIO()
    img.copy().convert("RGB").save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def render_page(**extra):
    ctx = {
        "linkedin": LINKEDIN_URL,
        "github": GITHUB_URL,
        "error_model": state["error"],
        "samples": get_samples(),
        "metrics": get_metrics(),
        "has_confusion": (BASE_DIR / "static" / "confusion_matrix.png").exists(),
        "has_comparison": (BASE_DIR / "static" / "model_comparison.png").exists(),
        "text_value": "",
        "message": None,
        "result": None,
        "preview": None,
        "sample_id": "",
    }
    ctx.update(extra)
    return render_template("index.html", **ctx)


def find_sample(sample_id):
    for s in get_samples():
        if str(s["id"]) == str(sample_id):
            return s
    return None


@app.route("/", methods=["GET"])
def index():
    sample = find_sample(request.args.get("sample", ""))
    if sample:
        img = Image.open(SAMPLE_DIR / f"{sample['id']}.jpg")
        return render_page(text_value=sample["text"], preview=to_data_uri(img), sample_id=sample["id"])
    return render_page()


@app.route("/predict", methods=["POST"])
def do_predict():
    if state["bundle"] is None:
        return render_page()

    text = request.form.get("text", "").strip()[:MAX_TEXT_CHARS]
    upload = request.files.get("image")
    sample_id = request.form.get("sample_id", "")

    img = None
    if upload and upload.filename:
        ext = upload.filename.rsplit(".", 1)[-1].lower() if "." in upload.filename else ""
        if ext not in ALLOWED_EXT:
            return render_page(text_value=text, message="Please upload a jpg, png, webp or gif image.")
        try:
            img = Image.open(upload.stream).convert("RGB")
        except Exception:
            return render_page(text_value=text, message="That file could not be read as an image.")
    elif sample_id and find_sample(sample_id):
        img = Image.open(SAMPLE_DIR / f"{sample_id}.jpg").convert("RGB")

    if not text and img is None:
        return render_page(message="Write a tweet, add an image, or pick an example - ideally both.")

    result = predict(text, img)
    if result is None:
        return render_page(
            text_value=text,
            message="None of those words are known to the model and there is no image, so there is nothing to score.",
        )

    keep_sample = sample_id if (img is not None and not (upload and upload.filename)) else ""
    return render_page(
        text_value=text,
        result=result,
        preview=to_data_uri(img) if img is not None else None,
        sample_id=keep_sample,
    )


@app.errorhandler(413)
def too_big(_):
    return render_page(message="That image is too large (limit is 6 MB)."), 413


load_models()

if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
