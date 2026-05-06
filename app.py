from flask import Flask, render_template, request
from pipeline.prediction_pipeline import predict
from src.logger import get_logger
from src.custom_exception import CustomException

app = Flask(__name__)
logger = get_logger(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error  = None

    if request.method == "POST":
        try:
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
            logger.info(f"Prediction served: {result}")

        except CustomException as ce:
            error = str(ce)
            logger.error(f"CustomException in /: {error}")

        except Exception as e:
            error = "An unexpected error occurred. Please check your inputs."
            logger.error(f"Unexpected error in /: {e}")

    return render_template("index.html", result=result, error=error)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)