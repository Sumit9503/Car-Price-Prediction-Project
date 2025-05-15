import pandas as pd 
import numpy as np
import pickle as pk
import streamlit as st
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Load the trained AI model
model = pk.load(open('model.pkl', 'rb'))
scaler = pk.load(open('scaler.pkl', 'rb'))  # Load the scaler used during training

st.header('🚗 AI-Driven Car Price Prediction')

# Load dataset
cars_data = pd.read_csv('Cardetails.csv')

def get_brand_name(car_name):
    return car_name.split(' ')[0].strip()

cars_data['name'] = cars_data['name'].apply(get_brand_name)

# Ensure consistent brand mapping
brand_mapping = {brand: idx + 1 for idx, brand in enumerate(cars_data['name'].unique())}

# User inputs
name = st.selectbox('Select Car Brand', cars_data['name'].unique())
year = st.slider('Car Manufactured Year', 1994, 2024)
km_driven = st.slider('No of kms Driven', 11, 200000)
fuel = st.selectbox('Fuel type', cars_data['fuel'].unique())
seller_type = st.selectbox('Seller type', cars_data['seller_type'].unique())
transmission = st.selectbox('Transmission type', cars_data['transmission'].unique())
owner = st.selectbox('Owner type', cars_data['owner'].unique())
mileage = st.slider('Car Mileage (km/l)', 10, 40)
engine = st.slider('Engine CC', 700, 5000)
max_power = st.slider('Max Power (bhp)', 0, 200)
seats = st.slider('No of Seats', 2, 10)

if st.button("Predict Price"):
    input_data_model = pd.DataFrame(
        [[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
        columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']
    )

    # Encoding categorical data using same mapping as training
    input_data_model['name'] = input_data_model['name'].map(brand_mapping)  
    input_data_model['owner'].replace(['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'], [1,2,3,4,5], inplace=True)
    input_data_model['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1,2,3,4], inplace=True)
    input_data_model['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1,2,3], inplace=True)
    input_data_model['transmission'].replace(['Manual', 'Automatic'], [1,2], inplace=True)

    # Apply StandardScaler transformation before prediction
    input_data_scaled = scaler.transform(input_data_model)

    # AI model prediction
    car_price = model.predict(input_data_scaled)[0]
    st.success(f'💰 Predicted Car Price: ₹{car_price:,.2f}')
# AI-driven analysis (Feature Importance)
    st.subheader('🔍 AI Insights: Feature Importance')
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({'Feature': input_data_model.columns, 'Importance': model.feature_importances_})
        st.bar_chart(importance_df.set_index('Feature'))
    else:
        st.write('Feature importance analysis not available for this model.')
