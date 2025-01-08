# -*- coding: utf-8 -*-
from keras.models import load_model
import streamlit as st
import joblib
import numpy as np

# Load the saved model
model = load_model('C:/Users/HP/Desktop/car_purchase_amount_model.keras')
scaler = joblib.load('C:/Users/HP/Desktop/scaler.pkl')
st.title('car_purchase_amount_model')

col1 , col2 ,col3 = st.columns(3)
with col1:
   gender = st.text_input('Gender')
with col2:
   age = st.text_input('Age')
with col3:
   salary = st.text_input('Annual Salary')
debt = st.text_input('Credit Card Debt')
net = st.text_input('Net Worth'  )



if st.button('Predict'):
    user_input = np.array([
           float(gender), float(age), float(salary), float(debt),float(net)
       ])
    
    input_data = scaler.transform([user_input])
    prediction = model.predict(input_data)
    st.write(f"Predicted Amount: ${prediction[0][0]:,.2f}")





























