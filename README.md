# CropCast AI

AI-powered crop production prediction web app built with Flask, scikit-learn, and XGBoost.

CropCast AI predicts production (in tonnes) from location, crop, season, year, and area, then returns:
- Predicted production
- Yield per hectare
- Confidence score
- Model name
- Human-readable interpretation

## 1. Project Overview

This project has two main parts:
- Model training pipeline in [train_model.py](train_model.py)
- Prediction web application in [app.py](app.py)

The frontend page in [templates/index.html](templates/index.html) and [static/js/main.js](static/js/main.js) calls backend APIs to:
- Populate dropdowns (states, districts, crops, seasons)
- Submit prediction requests
- Render feature-importance visualization

## 2. Features

- Crop production prediction for Indian agriculture data
- Input validation on both frontend and backend
- Auto-populated districts based on selected state
- Feature-importance chart for model explainability
- Confidence and interpretation for each prediction
- Clean single-page UI for form + results

## 3. Tech Stack

- Backend: Flask, Flask-CORS
- ML/Data: pandas, numpy, scikit-learn, XGBoost, joblib
- Frontend: HTML, CSS, Vanilla JavaScript, Chart.js

Dependency versions are listed in [requirements.txt](requirements.txt).

## 4. Project Structure

```text
Crop product prediction/
|- app.py
|- train_model.py
|- requirements.txt
|- README.md
|- data/
|  |- crop_production.csv
|- model/
|  |- feature_importance.json
|  |- crop_model.pkl          # generated locally after training
|  |- encoders.pkl            # generated locally after training
|- static/
|  |- css/
|  |  |- style.css
|  |- js/
|     |- main.js
|- templates/
	 |- index.html
```

## 5. Data Requirements

Expected input dataset file:
- [data/crop_production.csv](data/crop_production.csv)

Minimum required columns used by the pipeline:
- State_Name
- District_Name
- Crop_Year
- Season
- Crop
- Area
- Production

Source used in this project:
- https://www.kaggle.com/datasets/imtkaggleteam/crop-production

## 6. Setup and Installation

### 6.1 Clone

```bash
git clone https://github.com/deekshu15/crop-production-prediction.git
cd crop-production-prediction
```

### 6.2 Create virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 6.3 Install dependencies

```bash
pip install -r requirements.txt
```

## 7. Train the Model

Run:

```bash
python train_model.py
```

What training does:
1. Loads and cleans data (trim strings, type conversion, remove invalid rows/outliers)
2. Encodes categorical columns with LabelEncoder
3. Scales features using StandardScaler
4. Trains 3 regressors:
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor
5. Selects best model by highest R2 score
6. Saves artifacts:
- [model/crop_model.pkl](model/crop_model.pkl)
- [model/encoders.pkl](model/encoders.pkl)
- [model/feature_importance.json](model/feature_importance.json)

## 8. Run the Application

```bash
python app.py
```

Open:
- http://localhost:5000

## 9. API Documentation

### 9.1 GET /

Returns main web page.

### 9.2 POST /predict

Predict crop production.

Request JSON:

```json
{
	"state": "Karnataka",
	"district": "BENGALURU",
	"year": 2014,
	"season": "Kharif",
	"crop": "Rice",
	"area": 120.5
}
```

Success response:

```json
{
	"prediction": 356.21,
	"yield_per_hectare": 2.96,
	"confidence": 87,
	"model_used": "XGBoost Regressor",
	"interpretation": "Above average yield expected for this region",
	"unit": "tonnes"
}
```

Validation behavior:
- Returns 400 if area <= 0
- Returns 400 for unseen state/district/season/crop values
- Returns 500 for unexpected runtime errors

### 9.3 GET /api/stats

Returns metadata used by UI:
- states_list, seasons_list, crops_list
- districts_by_state
- total_records
- year_range
- feature_importance
- production summary arrays

### 9.4 GET /api/districts/<state>

Returns districts for selected state.

Response:

```json
{
	"districts": ["DISTRICT_A", "DISTRICT_B"]
}
```

## 10. Frontend Behavior

Implemented in [static/js/main.js](static/js/main.js):

1. On page load:
- Calls /api/stats
- Populates state/crop/season dropdowns
- Updates key counters
- Draws feature-importance chart

2. On state change:
- Calls /api/districts/<state>
- Populates district dropdown

3. On prediction submit:
- Validates fields
- Sends POST /predict
- Displays result card with confidence and interpretation

## 11. Model Details

Training features used by the model:
- Encoded State_Name
- Encoded District_Name
- Crop_Year
- Encoded Season
- Encoded Crop
- Area

Target:
- Production

Feature importance is derived from:
- feature_importances_ for tree-based models
- absolute coefficient values for linear model

## 12. Error Handling and Validation

Backend:
- File existence checks at startup for dataset and model artifacts
- Explicit unseen-category protection in safe_encode
- Structured JSON error responses

Frontend:
- Required field checks
- Area > 0 validation
- User-friendly error toast messages

## 13. Common Issues and Fixes

### App exits immediately with code 1

Cause:
- Missing model artifacts or missing dataset.

Fix:
1. Ensure [data/crop_production.csv](data/crop_production.csv) exists
2. Run: python train_model.py
3. Confirm [model/crop_model.pkl](model/crop_model.pkl) and [model/encoders.pkl](model/encoders.pkl) are created
4. Start again: python app.py

### GitHub push fails due large files

Cause:
- Model binaries can exceed GitHub file size limits.

Fix options:
1. Keep .pkl files out of git and generate locally
2. Use Git LFS for model binaries

## 14. Development Notes

- The project currently uses Flask debug mode in [app.py](app.py).
- CORS is enabled via Flask-CORS.
- Dashboard view was removed; only single-page predictor remains.

## 15. Suggested Improvements

1. Add automated tests for API routes
2. Add model versioning metadata
3. Add Dockerfile for reproducible deployment
4. Add production WSGI server config (gunicorn/waitress)
5. Add CI workflow for lint/test checks

## 16. License

No explicit license file is currently included. Add a LICENSE file before broad public distribution.
