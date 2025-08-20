from flask import Blueprint, request, jsonify
import numpy as np
from model import build_and_train_model
from preprocessing import load_and_preprocess_data

predict_route = Blueprint('predict_route', __name__)

# Load and preprocess data
X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_and_preprocess_data("diabetes.csv")

# Build and train the model
model = build_and_train_model(X_train, y_train, X_val, y_val)

@predict_route.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = np.array([data['pregnancies'], data['glucose'], data['bloodPressure'], data['skinThickness'],
                       data['insulin'], data['bmi'], data['diabetesPedigreeFunction'], data['age']])

    input_data_scaled = scaler.transform([input_data])
    prediction = model.predict(input_data_scaled)
    prediction_class = (prediction > 0.5).astype(int)

    return jsonify({
        'prediction': float(prediction[0][0]),
        'prediction_class': int(prediction_class[0][0])
    })
