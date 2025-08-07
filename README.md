# 🌾 Crop Recommendation System

A machine learning–powered Streamlit web application that recommends the most suitable crop to grow based climatic conditions of state capital districts across India.

---

## 📌 Overview

This project uses weather conditions, and rainfall statistics to predict the best crop using multiple ML models and ensemble voting. The app provides crop recommendations along with a relevant image, making it intuitive and user-friendly.

---

## 🚀 Features

- 🌡️ Scales climate features for improved accuracy
- 🧠 Uses ensemble of ML models (Random Forest, KNN, SVM, Logistic Regression, XGBoost)
- 📸 Displays crop image after prediction
- 🌍 Easy selection by state and district
- 📊 Shows predictions from individual models 
- 🔎 Clean and responsive Streamlit interface

---

## 🗂️ Project Structure

```
Crop_Recommendation_System/
├── Data/
│   ├── Raw/
│   │   └── Crop_recommendation.csv
│   └── Cleaned/
│       ├── Cleaned_dataset.csv
│       ├── Label_numbers.csv
│       └── state_capital_crop_data.csv
│
├── Notebooks/
│   ├── Data_Cleaning_and_EDA.ipynb
│   ├── Model_Training.ipynb
│   └── Prediction.ipynb
│
├── Models/
│   ├── Logistic Regression.pkl
│   ├── Naive Bayes.pkl
│   ├── Random Forest.pkl
│   ├── SVC.pkl
│   └── XGBoost.pkl
│
├── Scaler/
│   └── scaler.pkl
│
├── Streamlit_app/
│   ├── app.py
│   ├── Crop_images/         # Images for predicted crops
│   ├── District_data/       # Processed district/state crop data
│   ├── Saved_models/        # Models used in Streamlit
│   ├── Scaler/              # Scaler used in Streamlit
│   └── Utils/               # Helper Python notebooks
│
├── requirements.txt
└── README.md
```



