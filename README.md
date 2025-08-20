# GlucoGuard-diabetes-predictor

#  GlucoGuard: Diabetes Risk Predictor using Multilayer Perceptron

**GlucoGuard** is a machine learning-based diagnostic tool designed to predict the likelihood of diabetes onset using patient health metrics. Built with a robust Multilayer Perceptron (MLP) architecture, this project blends clinical insight with computational precision to deliver accurate, explainable predictions.

## Project Highlights

-  **Model**: Deep MLP with dropout, batch normalization, and ReLU activations
-  **Dataset**: Pima Indians Diabetes Dataset (Kaggle)
-  **Features**: BMI, glucose level, insulin, age, pregnancies, and engineered ratios
-  **Evaluation**: Accuracy, ROC-AUC, confusion matrix, and SHAP-based interpretability
-  **Deployment**: Flask API + TensorFlow Lite for mobile integration

## Why GlucoGuard?

Unlike generic predictors, GlucoGuard emphasizes:
- **Feature engineering** for enhanced clinical relevance
- **Model explainability** using SHAP values
- **Real-time simulation** for dynamic glucose monitoring (optional extension)
- **Modular design** for easy integration with wearables or EHR systems

##  Tech Stack

- Python, NumPy, Pandas, Scikit-learn
- TensorFlow / Keras
- SHAP for model interpretability
- Flask / FastAPI for deployment
- TensorFlow Lite (optional mobile support)

