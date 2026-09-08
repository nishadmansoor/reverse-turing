"""
Reverse Turing Test — Flask App
===============================
Serves the three classifiers plus the stacked ensemble.

Startup contract
----------------
This app loads only prebuilt artifacts and model weights. It does NOT read the
raw dataset and does NOT train anything — run `python src/build_artifacts.py`
first. That keeps boot around a second or two and resident memory low enough to
deploy, and it means a missing dataset is a build-time error with a clear
message rather than an import-time crash.
"""

import json
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from flask import Flask, flash, jsonify, redirect, render_template, request, url_for

PROJECT_ROOT = Path(__file__).parent.resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import (  # noqa: E402
    FEATURE_NAMES,
    extract_features,
    meta_features,
    text_to_heatmap,
)
from src.model_defs import CNN  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = PROJECT_ROOT / "models"

MIN_TEXT_LENGTH = 20
MAX_TEXT_LENGTH = 20_000
BERT_MAX_TOKENS = 256


class ArtifactsMissing(RuntimeError):
    """Raised when the app is started before the build step has run."""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _require(path: Path, hint: str):
    if not path.exists():
        raise ArtifactsMissing(
            f"Missing {path.relative_to(PROJECT_ROOT)}.\n"
            f"  Fix: {hint}"
        )
    return path


def _load_pickle(path: Path, hint: str):
    with open(_require(path, hint), "rb") as f:
        return pickle.load(f)


def _load_json(path: Path, hint: str):
    with open(_require(path, hint)) as f:
        return json.load(f)


def load_runtime():
    """Load artifacts and model weights. Fast and allocation-light."""
    build_hint = "python src/build_artifacts.py"

    logger.info("Loading artifacts...")
    metadata = _load_json(ARTIFACTS_DIR / "metadata.json", build_hint)
    explorer = _load_json(ARTIFACTS_DIR / "explorer.json", build_hint)
    pca = _load_pickle(ARTIFACTS_DIR / "pca.pkl", build_hint)
    stylo_model = _load_pickle(MODELS_DIR / "stylometric_classifier.pkl", build_hint)
    meta_model = _load_pickle(
        MODELS_DIR / "meta_classifier.pkl",
        "restore models/meta_classifier.pkl (see src/train_meta.py)",
    )

    logger.info("Loading CNN...")
    cnn_model = CNN()
    cnn_model.load_state_dict(torch.load(
        _require(MODELS_DIR / "cnn_classifier.pt", "python src/retrain.py --skip-bert --skip-stylo"),
        map_location="cpu", weights_only=True,
    ))
    cnn_model.eval()

    logger.info("Loading transformer...")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    bert_dir = MODELS_DIR / "bert_classifier"
    if not (bert_dir / "model.safetensors").exists() and not (bert_dir / "pytorch_model.bin").exists():
        raise ArtifactsMissing(
            f"Missing transformer weights in {bert_dir.relative_to(PROJECT_ROOT)}.\n"
            "  Fix: python src/retrain.py  (or pull them via git-lfs)"
        )
    tokenizer = AutoTokenizer.from_pretrained(str(bert_dir))
    bert_model = AutoModelForSequenceClassification.from_pretrained(str(bert_dir))
    bert_model.eval()

    averages = metadata["feature_averages"]
    examples = _load_json(Path(__file__).parent / "examples.json", "restore app/examples.json")

    # Optional: the held-out evaluation report powers the Method & Results page.
    # Absent on a fresh build, so the page degrades to a "run this" hint.
    evaluation = None
    eval_path = PROJECT_ROOT / "results" / f"evaluation_{metadata.get('dataset')}.json"
    if eval_path.exists():
        with open(eval_path) as f:
            evaluation = json.load(f)
        logger.info("Loaded evaluation report: %s", eval_path.name)
    else:
        logger.warning("No evaluation report at %s", eval_path.relative_to(PROJECT_ROOT))

    logger.info("Ready — dataset=%s, explorer=%d rows, built %s",
                metadata.get("dataset"), metadata.get("explorer_rows"),
                metadata.get("built_at"))

    return {
        "metadata": metadata,
        "explorer": explorer,
        "pca": pca,
        "stylo_model": stylo_model,
        "meta_model": meta_model,
        "cnn_model": cnn_model,
        "tokenizer": tokenizer,
        "bert_model": bert_model,
        "human_avg": np.array(averages["human"]),
        "ai_avg": np.array(averages["ai"]),
        "examples": examples,
        "evaluation": evaluation,
    }


models = load_runtime()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _predict_bert(text: str) -> tuple:
    tokens = models["tokenizer"](
        text, max_length=BERT_MAX_TOKENS, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    with torch.no_grad():
        probs = torch.softmax(models["bert_model"](**tokens).logits, dim=1)
    ai_prob = float(probs[0][1])
    pred = int(ai_prob >= 0.5)
    return pred, ai_prob, float(probs[0][pred])


def _predict_stylo(features: list) -> tuple:
    arr = np.array(features).reshape(1, -1)
    proba = models["stylo_model"].predict_proba(arr)[0]
    ai_prob = float(proba[1])
    pred = int(ai_prob >= 0.5)
    return pred, ai_prob, float(proba[pred])


def _predict_cnn(text: str) -> tuple:
    heatmap = text_to_heatmap(text)
    tensor = torch.tensor(heatmap, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    with torch.no_grad():
        probs = torch.softmax(models["cnn_model"](tensor), dim=1)
    ai_prob = float(probs[0][1])
    pred = int(ai_prob >= 0.5)
    return pred, ai_prob, float(probs[0][pred])


def _ensemble(text, bert_ai_prob, stylo_ai_prob, cv_ai_prob) -> tuple:
    """Combine the three models via the stacked meta-classifier.

    The meta-classifier is fitted on the validation split (see
    src/train_meta.py), so it learns how much to trust each base model from
    data the bases never trained on. An earlier version applied a hardcoded
    override — force "AI" whenever stylometry was >=0.7 confident and the
    transformer disagreed — to patch around a transformer that turned out to
    be barely trained. With the bases retrained and the stacker fitted
    honestly, that heuristic is no longer needed.

    Returns (verdict, confidence_pct, ai_probability).
    """
    features = meta_features(bert_ai_prob, stylo_ai_prob, cv_ai_prob, text)
    proba = models["meta_model"].predict_proba(features)[0]
    ai_prob = float(proba[1])
    verdict = "AI" if ai_prob >= 0.5 else "Human"
    return verdict, round(max(ai_prob, 1 - ai_prob) * 100, 1), ai_prob


def classify(text: str) -> dict:
    """Run the full pipeline over one text and build the template payload."""
    bert_pred, bert_ai_prob, bert_conf = _predict_bert(text)

    features = extract_features(text)
    stylo_pred, stylo_ai_prob, stylo_conf = _predict_stylo(features)

    cv_pred, cv_ai_prob, cv_conf = _predict_cnn(text)

    verdict, verdict_conf, verdict_ai_prob = _ensemble(
        text, bert_ai_prob, stylo_ai_prob, cv_ai_prob
    )

    feature_comparison = [
        {
            "name": name,
            "user_val": round(features[i], 2),
            "human_avg": round(float(models["human_avg"][i]), 2),
            "ai_avg": round(float(models["ai_avg"][i]), 2),
            "closer_to": (
                "ai"
                if abs(features[i] - models["ai_avg"][i]) < abs(features[i] - models["human_avg"][i])
                else "human"
            ),
        }
        for i, name in enumerate(FEATURE_NAMES)
    ]

    models_agree = len({bert_pred, stylo_pred, cv_pred}) == 1

    return {
        "text": text[:500],
        "text_truncated": len(text) > 500,
        "word_count": len(text.split()),
        "bert": "AI" if bert_pred == 1 else "Human",
        "bert_confidence": round(bert_conf * 100, 1),
        "bert_ai_prob": round(bert_ai_prob * 100, 1),
        "stylometric": "AI" if stylo_pred == 1 else "Human",
        "stylo_confidence": round(stylo_conf * 100, 1),
        "stylo_ai_prob": round(stylo_ai_prob * 100, 1),
        "opencv": "AI" if cv_pred == 1 else "Human",
        "opencv_confidence": round(cv_conf * 100, 1),
        "opencv_ai_prob": round(cv_ai_prob * 100, 1),
        "ensemble": verdict,
        "ensemble_confidence": verdict_conf,
        "ensemble_ai_prob": round(verdict_ai_prob * 100, 1),
        "models_agree": models_agree,
        "dissenters": [
            name for name, pred in
            [("BERT", bert_pred), ("Stylometric", stylo_pred), ("CNN", cv_pred)]
            if pred != int(verdict == "AI")
        ],
        "features": feature_comparison,
        "sankey": _build_sankey(bert_pred, stylo_pred, cv_pred, verdict),
    }


# Validated against the dark chart surface (see app/templates/base.html).
# This pair clears every colour-vision gate all-pairs; per-model hues did not,
# so model identity is carried by the node labels instead of by colour.
HUMAN_COLOR = "#199e70"
AI_COLOR = "#9085e9"
NEUTRAL_COLOR = "#898781"


def _build_sankey(bert_pred, stylo_pred, cv_pred, verdict) -> dict:
    """Sankey data for one text flowing through the three models.

    Colour encodes the *prediction being carried*, not which model produced
    it — the node labels already name the models, so spending hue on identity
    would waste the only free channel. A dissent is then legible as an
    off-colour ribbon flowing into the verdict node.
    """
    is_ai = int(verdict == "AI")

    def tone(pred, alpha=None):
        base = AI_COLOR if pred else HUMAN_COLOR
        if alpha is None:
            return base
        rgb = tuple(int(base[i:i + 2], 16) for i in (1, 3, 5))
        return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})"

    labels = [
        "Your text",
        f"BERT · {'AI' if bert_pred else 'Human'}",
        f"Stylometric · {'AI' if stylo_pred else 'Human'}",
        f"Visual CNN · {'AI' if cv_pred else 'Human'}",
        f"Ensemble · {verdict}",
    ]

    sources = [0, 0, 0, 1, 2, 3]
    targets = [1, 2, 3, 4, 4, 4]
    preds = [bert_pred, stylo_pred, cv_pred, bert_pred, stylo_pred, cv_pred]

    return {
        "labels": labels,
        "sources": sources,
        "targets": targets,
        "values": [1] * 6,
        # Inbound ribbons are neutral (the text has no verdict yet); outbound
        # ribbons take the colour of the prediction each model contributed.
        "link_colors": [
            "rgba(137, 135, 129, 0.22)" if s == 0 else tone(p, 0.45)
            for s, p in zip(sources, preds)
        ],
        "node_colors": [
            NEUTRAL_COLOR,
            tone(bert_pred), tone(stylo_pred), tone(cv_pred), tone(is_ai),
        ],
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.context_processor
def inject_metadata():
    """Make build metadata available to every template, including the base
    layout's footer and the error pages."""
    return {"metadata": models["metadata"]}


@app.route("/")
def home():
    return render_template("index.html", examples=models["examples"])


@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("text", "").strip()

    if not text:
        flash("Please enter some text to classify.", "error")
        return redirect(url_for("home"))

    if len(text) < MIN_TEXT_LENGTH:
        flash(f"Text too short — {len(text)} characters. "
              f"Please enter at least {MIN_TEXT_LENGTH} for a reliable result.", "error")
        return redirect(url_for("home"))

    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH]
        flash(f"Text truncated to {MAX_TEXT_LENGTH:,} characters.", "notice")

    return render_template("results.html", results=classify(text))


@app.route("/explore")
def explore():
    return render_template(
        "explore.html",
        metadata=models["metadata"],
        var_explained=models["metadata"]["pca_variance_explained"],
    )


@app.route("/about")
def about():
    return render_template(
        "about.html",
        metadata=models["metadata"],
        evaluation=models["evaluation"],
    )


@app.route("/api/explorer-data")
def api_explorer_data():
    return jsonify(models["explorer"])


@app.route("/api/pca-transform", methods=["POST"])
def api_pca_transform():
    """Project user text into the explorer's PCA space."""
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    if len(text) < MIN_TEXT_LENGTH:
        return jsonify({"error": f"Need at least {MIN_TEXT_LENGTH} characters."}), 400

    features = np.array(extract_features(text)).reshape(1, -1)
    coords = models["pca"].transform(features)
    return jsonify({"x": float(coords[0, 0]), "y": float(coords[0, 1])})


@app.route("/api/classify", methods=["POST"])
def api_classify():
    """JSON classification endpoint, so the demo is scriptable."""
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    if len(text) < MIN_TEXT_LENGTH:
        return jsonify({"error": f"Need at least {MIN_TEXT_LENGTH} characters."}), 400

    result = classify(text[:MAX_TEXT_LENGTH])
    result.pop("sankey", None)
    return jsonify(result)


@app.route("/health")
def health():
    """Liveness probe for the host platform."""
    return jsonify({
        "status": "ok",
        "dataset": models["metadata"].get("dataset"),
        "artifacts_built_at": models["metadata"].get("built_at"),
    })


@app.errorhandler(404)
def not_found(_):
    return render_template("error.html", code=404,
                           message="That page doesn't exist."), 404


@app.errorhandler(500)
def server_error(err):
    logger.exception("Unhandled error: %s", err)
    return render_template("error.html", code=500,
                           message="Something broke while classifying. Try a different text."), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", host="0.0.0.0", port=port)
