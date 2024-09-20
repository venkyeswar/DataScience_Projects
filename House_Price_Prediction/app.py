import streamlit as st
import pandas as pd
import joblib

# Load the scaler and model
model = joblib.load("model.pkl")
scaler = joblib.load("scaler_new.pkl")

# Function to predict house price
def predict_price(inputs):
    # Convert inputs to a DataFrame
    inputs_df = pd.DataFrame([inputs], columns=[
        'number_of_bedrooms', 'number_of_bathrooms', 'living_area', 'waterfront_present',
        'number_of_views', 'grade_of_the_house', 'area_of_house_excluding_basement',
        'Area_of_the_basement', 'Built_Year', 'Lattitude', 'Longitude'
    ])
    
    # Scale the inputs
    inputs_scaled = scaler.transform(inputs_df)

    # Get the prediction from the model
    prediction = model.predict(inputs_scaled)
    return prediction[0]

# Streamlit app starts here
st.title("House Price Prediction")

# Input fields for the model
number_of_bedrooms = st.number_input('Number of Bedrooms', min_value=1, value=1, step=1)
number_of_bathrooms = st.number_input('Number of Bathrooms', min_value=1, value=1, step=1)
living_area = st.number_input('Living Area (sqft)', min_value=100, value=1000, step=50)
waterfront_present = st.selectbox('Is Waterfront Present?', [0, 1])
number_of_views = st.number_input('Number of Views', min_value=0, value=0, step=1)
grade_of_the_house = st.number_input('Grade of the House (1-10)', min_value=1, max_value=10, value=1, step=1)
area_of_house_excluding_basement = st.number_input('Area of the House excluding Basement (sqft)', min_value=100, value=1000, step=50)
area_of_basement = st.number_input('Area of the Basement (sqft)', min_value=0, value=0, step=50)
build_year = st.number_input('Build Year', min_value=1900, value=2000, step=1)
latitude = st.number_input('Latitude', min_value=-90.0, max_value=90.0, value=53.0)
longitude = st.number_input('Longitude', min_value=-180.0, max_value=180.0, value=-114.0)

# Prediction button
if st.button("Predict House Price"):
    # Prepare inputs
    inputs = {
        'number_of_bedrooms': number_of_bedrooms,
        'number_of_bathrooms': number_of_bathrooms,
        'living_area': living_area,
        'waterfront_present': waterfront_present,
        'number_of_views': number_of_views,
        'grade_of_the_house': grade_of_the_house,
        'area_of_house_excluding_basement': area_of_house_excluding_basement,
        'Area_of_the_basement': area_of_basement,
        'Built_Year': build_year,
        'Lattitude': latitude,
        'Longitude': longitude
    }
    
    # Get the prediction
    price = predict_price(inputs)
    
    # Display the prediction
    st.success(f"Predicted House Price: ${price:,.2f}")
