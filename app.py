
import streamlit as st
import pandas as pd
import joblib
import json

# Page Configuration
st.set_page_config(page_title="Netflix Churn Predictor", page_icon="🎬", layout="centered")

# 1. Load Model & Data (Cached for performance)
@st.cache_resource
def load_resources():
    model = joblib.load('model.pkl')
    with open('columns.json', 'r') as f:
        data_info = json.load(f)
    return model, data_info

try:
    model, data_info = load_resources()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

cat_options = data_info['cat_columns']
num_cols = data_info['num_columns']

# 2. App Interface
st.title("🎬 Netflix Customer Churn Predictor")
st.markdown("Predict if a customer will cancel their subscription based on their usage patterns.")
st.divider()

# 3. Dynamic Inputs
user_input = {}
col1, col2 = st.columns(2)
all_columns = list(cat_options.keys()) + num_cols

for i, col_name in enumerate(all_columns):
    location = col1 if i % 2 == 0 else col2
    
    # Format labels for better reading (e.g., 'monthly_fee' -> 'Monthly Fee')
    label = col_name.replace('_', ' ').title()

    if col_name in cat_options:
        # Categorical: Dropdown
        user_input[col_name] = location.selectbox(label, cat_options[col_name])
    else:
        # Numerical: Number Input
        # We assume standard float input, min_value set to 0 to prevent negatives
        user_input[col_name] = location.number_input(label, value=0.0, min_value=0.0)

st.divider()

# 4. Prediction Logic
if st.button("🔮 Predict Churn Status", use_container_width=True):
    # Convert dictionary to DataFrame
    df = pd.DataFrame([user_input])
    
    try:
        # Get Probability and Prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        if prediction == 1:
            st.error(f"⚠️ **High Risk:** This customer is likely to CHURN.")
            st.metric(label="Churn Probability", value=f"{probability:.2%}")
        else:
            st.success(f"✅ **Safe:** This customer is likely to STAY.")
            st.metric(label="Churn Probability", value=f"{probability:.2%}")
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")
