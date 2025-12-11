
import streamlit as st
import pandas as pd
import joblib
import json

# 1. Load the model and data info
model = joblib.load('model.pkl')

with open('columns.json', 'r') as f:
    data_info = json.load(f)

cat_options = data_info['cat_columns']
num_cols = data_info['num_columns']

# 2. App Title
st.title("Netflix Customer Churn Predictor")
st.write("Enter customer details to predict if they will churn.")

# 3. Create Inputs dynamically
user_input = {}

# Create 2 columns for better layout
col1, col2 = st.columns(2)

all_columns = list(cat_options.keys()) + num_cols

# Iterate through columns to create widgets
for i, col_name in enumerate(all_columns):
    # Determine which column to place the widget in
    location = col1 if i % 2 == 0 else col2
    
    if col_name in cat_options:
        # It's categorical -> Create a Dropdown (Selectbox)
        user_input[col_name] = location.selectbox(f"Select {col_name}", cat_options[col_name])
    else:
        # It's numerical -> Create a Number Input
        user_input[col_name] = location.number_input(f"Enter {col_name}", value=0.0)

# 4. Predict Button
if st.button("Predict Churn"):
    # Convert input dict to DataFrame
    df = pd.DataFrame([user_input])
    
    # Predict
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    
    # Display Result
    if prediction == 1:
        st.error(f"⚠️ Churn Predicted! (Probability: {probability:.2%})")
        st.write("This customer is likely to cancel their subscription.")
    else:
        st.success(f"✅ No Churn Predicted (Probability: {probability:.2%})")
        st.write("This customer is likely to stay.")

