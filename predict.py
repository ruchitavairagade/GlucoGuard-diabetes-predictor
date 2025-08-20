import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import load_model  # Assuming you have saved the trained model
import pickle  # Import pickle for loading the scaler

# Function to load the model and scaler (trained earlier)
def load_trained_model_and_scaler():
    # Load the pre-trained model and scaler (make sure to save them after training)
    model = load_model("diabetes_model.h5")  # Ensure the model is saved as 'diabetes_model.h5' after training
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)  # Load the scaler using pickle
    return model, scaler

# Collect user input and make prediction
def predict_diabetes():
    # Load the trained model and scaler
    model, scaler = load_trained_model_and_scaler()

    # Features list as per the dataset columns
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    input_data = []

    print("Enter the following information:")

    # Collect user inputs for each feature
    for feature in features:
        while True:
            try:
                value = float(input(f"Enter {feature}: "))
                input_data.append(value)
                break
            except ValueError:
                print("Invalid input! Please enter a numeric value.")

    # Convert the input data into a DataFrame with the correct column names for prediction
    input_data_df = pd.DataFrame([input_data], columns=features)  # Create DataFrame with feature names

    # Standardize the input data using the trained scaler
    input_data_scaled = scaler.transform(input_data_df)  # Standardize the input data

    # Predict using the trained model
    prediction = model.predict(input_data_scaled)

    # Output the prediction
    if prediction > 0.5:
        print("Prediction: Diabetes (Yes)")
    else:
        print("Prediction: No Diabetes")

if __name__ == "__main__":
    predict_diabetes()
