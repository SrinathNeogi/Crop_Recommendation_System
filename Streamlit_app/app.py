import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from PIL import Image
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# --- Load Resources ---
scaler = joblib.load('Scaler/scaler.pkl')
label_map = pd.read_csv("Label_numbers.csv")
label_mapping = dict(zip(label_map['Number'], label_map['Name']))
district_df = pd.read_csv('District_data/state_capital_crop_data.csv')

# Define the feature columns
feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
district_df.dropna(subset=feature_cols, inplace=True)

# Load all saved models
model_folder = 'Saved_models'
models = {
    file.replace('.pkl', ''): joblib.load(os.path.join(model_folder, file))
    for file in os.listdir(model_folder) if file.endswith('.pkl')
}

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Crop Recommendation", page_icon="🌱")
st.title("🌾 Crop Recommendation System")
st.markdown("### Predict the best crop for your region based on soil and climate data.")

# --- User Input ---
state = st.selectbox("Select State", sorted(district_df['state'].unique()))
districts = district_df[district_df['state'] == state]['capital_district'].unique()
district = st.selectbox("Select District", sorted(districts))

if st.button("Predict Crop"):
    input_row = district_df[
        (district_df['state'].str.lower() == state.lower()) &
        (district_df['capital_district'].str.lower() == district.lower())
    ]

    if input_row.empty:
        st.error("🚫 No data found for the selected state and district.")
    else:
        input_features = input_row[feature_cols].values
        input_scaled = scaler.transform(input_features)

        # Model Predictions
        predictions = {}
        for model_name, model in models.items():
            pred = model.predict(input_scaled)[0]
            predictions[model_name] = pred

        # Majority Voting
        votes = Counter(predictions.values())
        top_label, vote_count = votes.most_common(1)[0]
        top_crop_name = label_mapping.get(top_label, "Unknown Crop")

        # --- Display Final Result ---
        st.markdown(
            f"<h2 style='text-align: center; color: green;'>🌾 Recommended Crop: <b>{top_crop_name.capitalize()}</b></h2>",
            unsafe_allow_html=True
        )

        # Show Crop Image
        image_path_jpg = f"Crop_images/{top_crop_name}.jpg"
        image_path_jpeg = f"Crop_images/{top_crop_name}.jpeg"
        if os.path.exists(image_path_jpg):
            st.image(image_path_jpg, caption=f"{top_crop_name.capitalize()}", use_container_width=True)
        elif os.path.exists(image_path_jpeg):
            st.image(image_path_jpeg, caption=f"{top_crop_name.capitalize()}", use_container_width=True)
        else:
            st.warning("🚫 No image available for this crop.")

        # --- Show All Model Predictions ---
        with st.expander("🔍 See predictions from each model"):
            for model_name, label in predictions.items():
                crop_name = label_mapping.get(label, "Unknown")
                st.write(f"**{model_name}** ➤ {crop_name} (Label: {label})")
