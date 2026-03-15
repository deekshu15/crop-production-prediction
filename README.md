# CropCast AI 🌾

> AI-powered crop production prediction using XGBoost trained on real Indian agriculture data

## Setup Instructions

### 1. Clone & Install
```bash
git clone <your-repo>
cd crop-prediction-app
pip install -r requirements.txt
```

### 2. Add Dataset
Place crop_production.csv inside the data/ folder.
Download from: https://www.kaggle.com/datasets/imtkaggleteam/crop-production

### 3. Train the Model
```bash
python train_model.py
```
This cleans the data, trains 3 models, picks the best one, and saves it.

### 4. Run the App
```bash
python app.py
```
Open browser: http://localhost:5000

## Features
- Predict crop production in tonnes for any state/crop/season combination
- Trained on 246,000+ real agricultural records
- XGBoost model with ~90%+ R² accuracy
- Fully responsive black + neon green UI

## Tech Stack
Python · Flask · XGBoost · scikit-learn · Chart.js · Vanilla JS
