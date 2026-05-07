import os
import joblib
import pandas as pd
from flask import Flask, render_template, request
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import MODEL_OUTPUT_PATH

app = Flask(__name__)
logger = get_logger(__name__)

SCALER_PATH = os.path.join("artifacts", "models", "scaler.pkl")
OHE_COLS    = ["cp", "slope", "thal"]

EXPECTED_COLS = [
    "age", "sex", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "ca",
    "cp_1", "cp_2", "cp_3",
    "slope_1", "slope_2",
    "thal_1", "thal_2", "thal_3",
]

def _load_artifacts():
    if not os.path.exists(MODEL_OUTPUT_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_OUTPUT_PATH}")

    _model = joblib.load(MODEL_OUTPUT_PATH)
    logger.info(f"Model loaded: {MODEL_OUTPUT_PATH}")

    _scaler = None
    if os.path.exists(SCALER_PATH):
        _scaler = joblib.load(SCALER_PATH)
        logger.info(f"Scaler loaded: {SCALER_PATH}")
    else:
        logger.warning(f"Scaler not found: {SCALER_PATH}")

    return _model, _scaler

try:
    model, scaler = _load_artifacts()
except Exception as e:
    logger.error(f"Artifact loading error: {e}")
    model, scaler = None, None

def predict(input_data: dict) -> str:
    try:
        df = pd.DataFrame([input_data])

        df = pd.get_dummies(df, columns=OHE_COLS, drop_first=True)

        for col in EXPECTED_COLS:
            if col not in df.columns:
                df[col] = 0
        df = df[EXPECTED_COLS]

        if scaler is not None:
            X = scaler.transform(df)
        else:
            logger.warning("Scaler not found, continuing with raw values.")
            X = df.values

        prediction = int(model.predict(X)[0])
        return "Heart Disease Detected" if prediction == 1 else "No Heart Disease"

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise CustomException("Error occurred during prediction.", e)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error  = None

    if request.method == "POST":
        try:
            if model is None:
                raise ValueError("Model not loaded. Please run the pipeline first.")

            input_data = {
                "age":      float(request.form["age"]),
                "sex":      int(request.form["sex"]),
                "cp":       int(request.form["cp"]),
                "trestbps": float(request.form["trestbps"]),
                "chol":     float(request.form["chol"]),
                "fbs":      int(request.form["fbs"]),
                "restecg":  int(request.form["restecg"]),
                "thalach":  float(request.form["thalach"]),
                "exang":    int(request.form["exang"]),
                "oldpeak":  float(request.form["oldpeak"]),
                "slope":    int(request.form["slope"]),
                "ca":       int(request.form["ca"]),
                "thal":     int(request.form["thal"]),
            }

            result = predict(input_data)
            logger.info(f"Prediction result: {result}")

        except CustomException as ce:
            error = str(ce)
            logger.error(f"CustomException: {error}")

        except Exception as e:
            error = f"Error: {str(e)}"
            logger.error(f"Unexpected error: {e}")

    return render_template("index.html", result=result, error=error)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)