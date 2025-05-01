import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

# Set page title
st.set_page_config(page_title="Breast Cancer Prediction")

# App header
st.title("🧬 Breast Cancer Prediction App")

# Feature names in order
column_list = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# Create a clean reference table
column_df = pd.DataFrame({
    'Feature #': list(range(1, 31)),
    'Feature Name': column_list
})

# Display the reference table
st.markdown("### Feature Reference Table")
st.markdown("Below is the list of features and the exact order in which you must paste the 30 standardized values:")
st.dataframe(column_df, use_container_width=True, hide_index=True)

# Input instructions
st.markdown("### 📥 Paste Standardized Input")
st.markdown("Paste **30 comma-separated standardized values** below, matching the feature order above:")

# Text input box
raw_input = st.text_area("Paste Input (comma-separated)", height=150)

# Predict button logic
if st.button("Predict"):
    try:
        # Convert input to float list
        values = [float(x.strip()) for x in raw_input.split(',')]
        
        if len(values) != 30:
            st.warning("❗ Please enter exactly 30 values.")
        else:
            input_array = np.array(values).reshape(1, -1)
            prediction = model.predict(input_array)[0]

            if prediction == 1:
                st.error("The tumor is **Cancerous (Malignant)**.")
            else:
                st.success("The tumor is **Not Cancerous (Benign)**.")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
