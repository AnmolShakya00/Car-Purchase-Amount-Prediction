Car Purchase Amount Prediction Model
Project Overview
This project implements a machine learning model that predicts the car purchase amount based on user input, including gender, age, annual salary, credit card debt, and net worth. The model is built using a Neural Network with Keras, and it is deployed as a Streamlit web application, allowing users to input their details and receive a prediction for the car purchase amount.

Features
Model: A Neural Network model built and trained using Keras for predicting car purchase amount.
User Input: Users provide details such as gender, age, salary, credit card debt, and net worth.
Prediction: After entering the details, the user can get an estimated car purchase amount.
Deployment: The model is deployed using Streamlit for easy access via a web interface.
Scaler: The data is preprocessed using a StandardScaler to normalize the input before prediction.
Requirements
To run this project locally, you need to install the following Python libraries:

1-streamlit
2-keras
3-tensorflow
4-joblib
5-numpy
6-scikit-learn
You can install the dependencies using pip:
pip install streamlit keras tensorflow joblib numpy scikit-learn

Files
car_purchase_amount_model.keras: The trained neural network model for predicting the car purchase amount.
scaler.pkl: The pre-fitted StandardScaler used for normalizing input data.
app.py: The Streamlit application script for interacting with the model.

Model Details
The model is a fully connected neural network built using Keras. It is trained on various features, including:

Gender: Encoded as 1 for male and 0 for female.
Age: The user's age in years.
Annual Salary: The user's annual salary in monetary value.
Credit Card Debt: The user's total credit card debt in monetary value.
Net Worth: The user's net worth in monetary value.
The model is designed to predict the expected car purchase amount based on these features.

Training Process
The model was trained using the following architecture:

Input Layer: 5 nodes corresponding to the 5 features.
Hidden Layers: Two fully connected layers with ReLU activation.
Output Layer: A single node for the predicted car purchase amount.
Optimizer: Adam optimizer.
Loss Function: Mean Squared Error (MSE).
Steps for Training:
Collect and preprocess the data.
Train the neural network on the data.
Save the trained model (car_purchase_amount_model.keras).
Save the fitted scaler (scaler.pkl).
