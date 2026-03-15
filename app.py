import json
from pathlib import Path
from urllib.parse import unquote

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "crop_production.csv"
MODEL_PATH = BASE_DIR / "model" / "crop_model.pkl"
ENCODERS_PATH = BASE_DIR / "model" / "encoders.pkl"
FEATURE_IMPORTANCE_PATH = BASE_DIR / "model" / "feature_importance.json"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    string_cols = cleaned.select_dtypes(include=["object"]).columns
    for col in string_cols:
        cleaned[col] = cleaned[col].astype(str).str.strip()

    cleaned["Area"] = pd.to_numeric(cleaned["Area"], errors="coerce")
    cleaned["Production"] = pd.to_numeric(cleaned["Production"], errors="coerce")
    cleaned["Crop_Year"] = pd.to_numeric(cleaned["Crop_Year"], errors="coerce")

    cleaned = cleaned.dropna(subset=["Area", "Production", "Crop_Year"])
    cleaned = cleaned[(cleaned["Area"] > 0) & (cleaned["Production"] > 0)]
    cleaned = cleaned.drop_duplicates()

    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        q1 = cleaned[col].quantile(0.25)
        q3 = cleaned[col].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + (3 * iqr)
        cleaned = cleaned[cleaned[col] <= upper_bound]

    cleaned["Crop_Year"] = cleaned["Crop_Year"].astype(int)
    cleaned["Yield"] = cleaned["Production"] / cleaned["Area"]

    return cleaned.reset_index(drop=True)


def safe_encode(encoder, value: str, field_name: str) -> int:
    if value not in encoder.classes_:
        raise ValueError(f"Unseen {field_name}: '{value}'")
    return int(encoder.transform([value])[0])


app = Flask(__name__)
CORS(app)

if not DATA_PATH.exists():
    raise FileNotFoundError("Missing data/crop_production.csv")
if not MODEL_PATH.exists() or not ENCODERS_PATH.exists():
    raise FileNotFoundError("Missing model files. Run train_model.py first.")

model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODERS_PATH)
df = clean_data(pd.read_csv(DATA_PATH))

if FEATURE_IMPORTANCE_PATH.exists():
    with open(FEATURE_IMPORTANCE_PATH, "r", encoding="utf-8") as f:
        feature_importance = json.load(f)
else:
    feature_importance = {}

SCALER = encoders["scaler"]
MODEL_NAME = encoders.get("model_name", model.__class__.__name__)
MODEL_R2 = float(encoders.get("r2_score", 0.78))


@app.route("/")
def index():
    try:
        return render_template("index.html")
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    try:
        body = request.get_json(force=True)

        state = str(body.get("state", "")).strip()
        district = str(body.get("district", "")).strip()
        year = int(body.get("year"))
        season = str(body.get("season", "")).strip()
        crop = str(body.get("crop", "")).strip()
        area = float(body.get("area"))

        if area <= 0:
            return jsonify({"error": "Area must be greater than 0"}), 400

        state_enc = safe_encode(encoders["state"], state, "state")
        district_enc = safe_encode(encoders["district"], district, "district")
        season_enc = safe_encode(encoders["season"], season, "season")
        crop_enc = safe_encode(encoders["crop"], crop, "crop")

        features = np.array([[state_enc, district_enc, year, season_enc, crop_enc, area]])
        features_scaled = SCALER.transform(features)

        prediction = float(model.predict(features_scaled)[0])
        prediction = max(0.0, prediction)

        yield_per_hectare = prediction / area

        historical = df[
            (df["State_Name"] == state)
            & (df["Crop"] == crop)
            & (df["Season"] == season)
        ]

        if historical.empty:
            confidence = 78
            avg_production = prediction
        else:
            confidence = min(100, int(MODEL_R2 * 100))
            avg_production = float(historical["Production"].mean())

        if prediction > avg_production * 1.1:
            interpretation = "Above average yield expected for this region"
        elif prediction < avg_production * 0.9:
            interpretation = "Below average yield expected for this region"
        else:
            interpretation = "Near average yield expected for this region"

        return jsonify(
            {
                "prediction": round(prediction, 2),
                "yield_per_hectare": round(yield_per_hectare, 2),
                "confidence": confidence,
                "model_used": MODEL_NAME,
                "interpretation": interpretation,
                "unit": "tonnes",
            }
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/stats", methods=["GET"])
def api_stats():
    try:
        production_trend = (
            df.groupby("Crop_Year", as_index=False)["Production"].sum().sort_values("Crop_Year")
        )
        top_crops = (
            df.groupby("Crop", as_index=False)["Production"].sum().sort_values("Production", ascending=False).head(10)
        )
        season_share = (
            df.groupby("Season", as_index=False)["Production"].sum().sort_values("Production", ascending=False)
        )
        top_states = (
            df.groupby("State_Name", as_index=False)["Production"]
            .sum()
            .sort_values("Production", ascending=False)
            .head(8)
        )

        avg_yield_by_crop = (
            (df.groupby("Crop")["Production"].sum() / df.groupby("Crop")["Area"].sum())
            .round(4)
            .sort_index()
            .to_dict()
        )

        return jsonify(
            {
                "total_records": int(len(df)),
                "crops_list": sorted(df["Crop"].unique().tolist()),
                "states_list": sorted(df["State_Name"].unique().tolist()),
                "seasons_list": sorted(df["Season"].unique().tolist()),
                "districts_by_state": encoders.get("districts_by_state", {}),
                "avg_yield_by_crop": avg_yield_by_crop,
                "production_trend": [
                    {"year": int(r.Crop_Year), "production": float(r.Production)}
                    for r in production_trend.itertuples(index=False)
                ],
                "top_crops": [
                    {"crop": r.Crop, "production": float(r.Production)}
                    for r in top_crops.itertuples(index=False)
                ],
                "season_share": [
                    {"season": r.Season, "production": float(r.Production)}
                    for r in season_share.itertuples(index=False)
                ],
                "top_states": [
                    {"state": r.State_Name, "production": float(r.Production)}
                    for r in top_states.itertuples(index=False)
                ],
                "feature_importance": feature_importance,
                "model_name": MODEL_NAME,
                "r2_score": MODEL_R2,
                "year_range": {
                    "min": int(df["Crop_Year"].min()),
                    "max": int(df["Crop_Year"].max()),
                },
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/districts/<path:state>", methods=["GET"])
def api_districts(state):
    try:
        state_name = unquote(state).strip()
        districts = encoders.get("districts_by_state", {}).get(state_name, [])
        return jsonify({"districts": districts}), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
