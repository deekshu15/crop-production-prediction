import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor


DATA_PATH = Path("data/crop_production.csv")
MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "crop_model.pkl"
ENCODERS_PATH = MODEL_DIR / "encoders.pkl"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "feature_importance.json"


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


def build_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        values = np.array(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.abs(np.array(model.coef_, dtype=float).ravel())
    else:
        values = np.zeros(len(feature_names), dtype=float)

    if values.sum() > 0:
        values = values / values.sum()

    feature_map = {name: float(val) for name, val in zip(feature_names, values)}
    return dict(sorted(feature_map.items(), key=lambda x: x[1], reverse=True))


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. Place crop_production.csv in data/."
        )

    df = pd.read_csv(DATA_PATH)
    print("Loaded dataset shape:", df.shape)
    print("First 5 rows:")
    print(df.head())

    df = clean_data(df)

    le_state = LabelEncoder()
    le_district = LabelEncoder()
    le_season = LabelEncoder()
    le_crop = LabelEncoder()

    df["State_Name_Enc"] = le_state.fit_transform(df["State_Name"])
    df["District_Name_Enc"] = le_district.fit_transform(df["District_Name"])
    df["Season_Enc"] = le_season.fit_transform(df["Season"])
    df["Crop_Enc"] = le_crop.fit_transform(df["Crop"])

    districts_by_state = (
        df.groupby("State_Name")["District_Name"]
        .apply(lambda x: sorted(set(x.tolist())))
        .to_dict()
    )

    feature_columns = [
        "State_Name_Enc",
        "District_Name_Enc",
        "Crop_Year",
        "Season_Enc",
        "Crop_Enc",
        "Area",
    ]
    feature_names = [
        "State_Name",
        "District_Name",
        "Crop_Year",
        "Season",
        "Crop",
        "Area",
    ]

    X = df[feature_columns]
    y = df["Production"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=150, random_state=42, n_jobs=-1
        ),
        "XGBoost Regressor": XGBRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
        ),
    }

    scores = {}
    trained = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        mae = float(mean_absolute_error(y_test, pred))
        r2 = float(r2_score(y_test, pred))
        scores[name] = {"RMSE": rmse, "MAE": mae, "R2": r2}
        trained[name] = model

    print("\n| Model                  | RMSE         | MAE          | R²       |")
    print("|------------------------|--------------|--------------|----------|")
    for name, metric in scores.items():
        print(
            f"| {name:<22} | {metric['RMSE']:<12.4f} | {metric['MAE']:<12.4f} | {metric['R2']:<8.4f} |"
        )

    best_model_name = max(scores, key=lambda n: scores[n]["R2"])
    best_model = trained[best_model_name]
    best_r2 = scores[best_model_name]["R2"]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    encoders_dict = {
        "state": le_state,
        "district": le_district,
        "season": le_season,
        "crop": le_crop,
        "states_list": sorted(df["State_Name"].unique().tolist()),
        "districts_list": sorted(df["District_Name"].unique().tolist()),
        "seasons_list": sorted(df["Season"].unique().tolist()),
        "crops_list": sorted(df["Crop"].unique().tolist()),
        "districts_by_state": districts_by_state,
        "scaler": scaler,
        "model_name": best_model_name,
        "r2_score": best_r2,
    }

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(encoders_dict, ENCODERS_PATH)

    feature_importance = build_feature_importance(best_model, feature_names)
    with open(FEATURE_IMPORTANCE_PATH, "w", encoding="utf-8") as fp:
        json.dump(feature_importance, fp, indent=2)

    print("\nTraining Summary")
    print(f"Best model name: {best_model_name}")
    print(f"R² on test set: {best_r2:.4f}")
    print("Top 3 most important features:")
    for name, value in list(feature_importance.items())[:3]:
        print(f"- {name}: {value:.4f}")
    print(f"Total training samples used: {len(X_train)}")


if __name__ == "__main__":
    main()
