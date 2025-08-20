import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(filepath):
    # Load the dataset
    df = pd.read_csv(filepath)
    
    # Separate features and target
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    
    # Handle missing values
    X['Glucose'] = X['Glucose'].replace(0, np.nan).fillna(X['Glucose'].mean())
    X['BloodPressure'] = X['BloodPressure'].replace(0, np.nan).fillna(X['BloodPressure'].mean())
    X['SkinThickness'] = X['SkinThickness'].replace(0, np.nan).fillna(X['SkinThickness'].mean())
    X['Insulin'] = X['Insulin'].replace(0, np.nan).fillna(X['Insulin'].mean())
    X['BMI'] = X['BMI'].replace(0, np.nan).fillna(X['BMI'].mean())
    
    # Standardize the dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler
