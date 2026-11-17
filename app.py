import json, os, joblib, pandas as pd
from flask import Flask, jsonify, request, send_from_directory

BASE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(BASE, "artifacts")
app = Flask(__name__, static_folder="static", static_url_path="/static")

# Loaded once at startup - never retrained per request
SCALER = joblib.load(os.path.join(ART, "scaler.joblib"))
MODEL = joblib.load(os.path.join(ART, "rf.joblib"))
with open(os.path.join(ART, "metrics.json")) as f:
    METRICS = json.load(f)
FEATURES = METRICS["features"]  # exact training order

RANGES = {"Time_spent_Alone": (0, 11), "Social_event_attendance": (0, 10), "Going_outside": (0, 7),
          "Friends_circle_size": (0, 15), "Post_frequency": (0, 10)}
YES_NO = ("Stage_fear", "Drained_after_socializing")

def validate(data):
    if not isinstance(data, dict):
        return None, "Request body must be a JSON object."
    missing = [k for k in FEATURES if k not in data]
    if missing:
        return None, f"Missing fields: {', '.join(missing)}."
    row = {}
    for k in FEATURES:
        v = data[k]
        if k in YES_NO:
            if v not in ("Yes", "No"):
                return None, f"{k} must be 'Yes' or 'No'."
            row[k] = 1 if v == "Yes" else 0
        else:
            lo, hi = RANGES[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
                return None, f"{k} must be a number between {lo} and {hi}."
            row[k] = v
    return row, None

@app.after_request
def no_cache(resp):                      # always serve the latest HTML/CSS/JS while developing
    resp.headers["Cache-Control"] = "no-store"
    return resp

@app.get("/")
def index():
    return send_from_directory("static", "index.html")

@app.get("/api/metrics")
def metrics():
    return jsonify(METRICS)

@app.post("/api/predict")
def predict():
    row, err = validate(request.get_json(silent=True))
    if err:
        return jsonify({"error": err}), 400
    X = pd.DataFrame([row], columns=FEATURES)           # exact feature order
    p = float(MODEL.predict_proba(SCALER.transform(X))[0, 1])
    return jsonify({"prediction": "Extrovert" if p >= 0.5 else "Introvert",
                    "extrovert_probability": round(p, 4),
                    "introvert_probability": round(1 - p, 4)})

if __name__ == "__main__":
    app.run(debug=True, port=5002)
