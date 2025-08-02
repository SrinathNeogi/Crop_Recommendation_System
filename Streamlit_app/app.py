import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from PIL import Image
from collections import Counter
import warnings

# --- Streamlit Page Config ---
st.set_page_config(page_title="Crop Recommendation")

warnings.filterwarnings('ignore')

# --- Base Directory ---
base_path = os.path.abspath(os.path.dirname(__file__))

# --- Load Scaler ---
scaler = joblib.load(os.path.join(base_path, 'Scaler', 'scaler.pkl'))

# --- Load Label Mapping ---
label_map = pd.read_csv(os.path.join(base_path, "Label_numbers.csv"))
label_mapping = dict(zip(label_map['Number'], label_map['Name']))

# --- Load Dataset ---
district_df = pd.read_csv(os.path.join(base_path, 'District_data', 'state_capital_crop_data.csv'))
feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
district_df.dropna(subset=feature_cols, inplace=True)

# --- Load Models ---
model_folder = os.path.join(base_path, 'Saved_models')
models = {
    file.replace('.pkl', ''): joblib.load(os.path.join(model_folder, file))
    for file in os.listdir(model_folder) if file.endswith('.pkl')
}

# --- Streamlit UI ---
st.title("Crop Recommendation System")
st.subheader("Predict the best crop based on your region's data.")

# --- User Input ---
state = st.selectbox("Select State", sorted(district_df['state'].unique()))
districts = sorted(district_df[district_df['state'] == state]['capital_district'].unique())
district = st.selectbox("Select District", districts)

if st.button("Predict Crop"):
    input_row = district_df[
        (district_df['state'].str.lower() == state.lower()) &
        (district_df['capital_district'].str.lower() == district.lower())
    ]

    if input_row.empty:
        st.error("No data found for the selected state and district.")
    else:
        input_features = input_row[feature_cols].values
        input_scaled = scaler.transform(input_features)

        # Predict using all models
        predictions = {
            model_name: model.predict(input_scaled)[0]
            for model_name, model in models.items()
        }

        # Majority Voting
        top_label = Counter(predictions.values()).most_common(1)[0][0]
        top_crop_name = label_mapping.get(top_label, "Unknown Crop")

        # Output
        st.markdown(
            f"<h2 style='text-align: center; color: green;'>Recommended Crop: <b>{top_crop_name}</b></h2>",
            unsafe_allow_html=True
        )

        # Image
        image_folder = os.path.join(base_path, "Crop_images")
        for ext in ['jpg', 'jpeg', 'png']:
            image_path = os.path.join(image_folder, f"{top_crop_name}.{ext}")
            if os.path.exists(image_path):
                st.image(image_path, caption=top_crop_name, use_container_width=True)
                break
        else:
            st.warning("Image not available for this crop.")

        # Model votes
        with st.expander("Model-wise Predictions"):
            for model_name, label in predictions.items():
                crop_name = label_mapping.get(label, "Unknown")
                st.write(f"{model_name}: {crop_name} (Label: {label})")
