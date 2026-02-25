"""
Flask API for Rainfall Prediction (Rain Tomorrow).
Loads the trained LightGBM model and serves predictions.
"""

import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS

app = Flask(__name__, static_folder="templates/static", template_folder="templates")
CORS(app)

# Feature order must match training (X.select_dtypes(exclude=['object']).columns)
FEATURE_NAMES = [
    "MinTemp", "MaxTemp", "Rainfall", "WindGustSpeed",
    "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm",
    "Pressure9am", "Pressure3pm", "Temp9am", "Temp3pm", "RainToday"
]

model = None
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "rainfall_model.pkl")


def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        return True
    return False


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Run the notebook to generate rainfall_model.pkl first."}), 503

    try:
        data = request.get_json() or {}
        values = []
        for name in FEATURE_NAMES:
            v = data.get(name)
            if v is None or v == "":
                return jsonify({"error": f"Missing or invalid value for: {name}"}), 400
            try:
                values.append(float(v))
            except (TypeError, ValueError):
                return jsonify({"error": f"Invalid number for: {name}"}), 400

        X = np.array([values])
        pred = model.predict(X)[0]
        proba = None
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X)[0][1])  # P(Rain Tomorrow = Yes)

        # Rainfall intensity categories
        intensity = "Light"
        suggestion = "Normal activities with caution"
        if proba is not None:
            pct = proba * 100
            if pct >= 67:
                intensity = "Heavy"
                suggestion = "Protect crops & livestock"
            elif pct >= 34:
                intensity = "Moderate"
                suggestion = "Avoid fertilizer spraying"

        # Render appropriate template
        if pred == 1:
            return render_template(
                "chance.html",
                probability=round(proba * 100, 2),
                intensity=intensity,
                suggestion=suggestion
            )
        else:
            return render_template(
                "noChance.html",
                intensity=intensity,
                suggestion=suggestion
            )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/features", methods=["GET"])
def features():
    return jsonify({"features": FEATURE_NAMES})


@app.route("/")
def index():
    return send_from_directory(app.template_folder, "index.html")


@app.route("/static/<path:path>")
def static_files(path):
    return send_from_directory(app.static_folder, path)


if __name__ == "__main__":
    load_model()
    if model is None:
        print("Warning: rainfall_model.pkl not found. Run the notebook to train and save the model.")
    app.run(host="0.0.0.0", port=5000, debug=True)