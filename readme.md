# Diabetes Prediction Using Multilayer Perceptrons

### Overview
This project uses a Multilayer Perceptron (MLP) neural network to predict the likelihood of diabetes in individuals based on various health metrics. The model is trained on a dataset of health metrics to classify whether a person has diabetes or not.### Diabetes Predection Using MLP
Diabetes Prediction Using Multilayer Perceptrons
Overview
This project aims to predict the likelihood of diabetes in individuals using a Multilayer Perceptron (MLP), a type of artificial neural network. The model is trained on a dataset of health metrics to classify whether a person has diabetes or not.


### Flask API
The Flask API provides an endpoint to make predictions based on user input. It uses an MLP model trained on health metrics to classify whether a person has diabetes or not.


### Model Training
**The MLP model is designed with:**
* Input Layer: 8 neurons (one for each feature)
* Hidden Layer 1: 32 neurons with ReLU activation
* Hidden Layer 2: 16 neurons with ReLU activation
* Output Layer: 1 neuron with sigmoid activation


### React Frontend
The React frontend allows users to input health metrics and view prediction results. It interacts with the Flask API to fetch predictions.
* React app :
![react_app](./assets/diabetes.png)
* Results in the react app
![react_app_result](./assets/diabetes_results.png)


### Data Visualisation
* Distribution of Key Features in Diabetes Prediction Dataset
![data](./assets/data.png)
* Density Plots of Features for Diabetes Prediction by Outcome
![outcome](./assets/output.png)
* Confusion Matrix for Diabetes Prediction Model
![matrix](./assets/matrix.png)
* ROC Curve for Diabetes Prediction Model
![curve](./assets/curve.png)


### Running The Predictor
To use the diabetes prediction model, follow these steps:

* **Data Analysis and Model Training**
1. Run the following command to execute the script that performs data visualization and model training:
   `python main.py`

2. This script will:
    Display histograms of the dataset’s features.
    1. Show density plots comparing the distribution of each feature for diabetic and non-diabetic cases.
    2. Replace zero values in critical columns with NaN and then fill missing values with the column mean.
    3. Standardize the features and split the data into training, validation, and test sets.
    4. Build and train an MLP model with the specified architecture.
    5. Evaluate the model’s performance and print accuracy metrics.
    6. Generate and display a confusion matrix and ROC curve to assess the model’s predictive performance.


* **Running the Flask API and React App**
1. Open a terminal and start the Flask API server with:
    `python app.py`
2. In a separate terminal, navigate to the React app directory and start the React development server:
    1. `cd diabetes-prediction-app`
    2. `npm start`
