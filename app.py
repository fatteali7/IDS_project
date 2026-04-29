from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    features = np.array(data["features"]).reshape(1, -1)

    # Scale input
    features = scaler.transform(features)

    prediction = model.predict(features)[0]
    confidence = model.predict_proba(features).max()

    return jsonify({
        "prediction": int(prediction),
        "confidence": float(confidence)
    })

if __name__ == "__main__":
    app.run(debug=True)