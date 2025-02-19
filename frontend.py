import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np
from keras.models import load_model

import pandas as pd

from models import logistic_regression


def predict_heart_disease():
    """
    Gathers input from the GUI, preprocesses it, and makes predictions using the Random Forest model.
    """
    try:
        # Get input values
        inputs = []
        for entry in input_entries:
            value = float(entry.get())
            inputs.append(value)

        # Convert inputs to a Pandas DataFrame with the correct column names
        input_data = pd.DataFrame([inputs], columns=[
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ])

        # Preprocess input
        features_processed = preprocessing_pipeline.transform(input_data)

        # Predict using the Random Forest model
        #prediction = random_forest_model.predict(features_processed)[0]

        # Predict using the Neural Network model
        probability = best_bb_network.predict(features_processed)[0][0]
        prediction = 1 if probability > 0.5 else 0

        # Display result
        result = "The patient is likely to have heart disease." if prediction == 1 else "The patient is unlikely to have heart disease."
        messagebox.showinfo("Prediction Result", result)
    except Exception as e:
        messagebox.showerror("Input Error", f"Invalid input: {str(e)}")


# Create the GUI
root = tk.Tk()
root.title("Heart Disease Prediction")
root.geometry("600x700")
root.configure(bg="#f0f4f7")

# Styling
label_font = ("Arial", 12, "bold")
entry_font = ("Arial", 12)
button_font = ("Arial", 14, "bold")

# Input fields
fields = [
    "Age",
    "Sex (1=Male, 0=Female)",
    "Chest Pain Type (0-3)",
    "Resting Blood Pressure",
    "Serum Cholesterol (mg/dl)",
    "Fasting Blood Sugar (1=Yes, 0=No)",
    "Resting ECG Results (0-2)",
    "Max Heart Rate Achieved",
    "Exercise Induced Angina (1=Yes, 0=No)",
    "Oldpeak (ST depression)",
    "Slope of Peak Exercise ST (0-2)",
    "Number of Major Vessels (0-3)",
    "Thalassemia (1-3)"
]

input_entries = []

for idx, field in enumerate(fields):
    label = tk.Label(root, text=field, bg="#f0f4f7", font=label_font)
    label.grid(row=idx, column=0, padx=10, pady=5, sticky="e")

    entry = tk.Entry(root, font=entry_font, relief="solid", bd=2)
    entry.grid(row=idx, column=1, padx=10, pady=5, ipadx=5, ipady=3)
    input_entries.append(entry)

# Prediction button
predict_button = tk.Button(
    root,
    text="Predict",
    command=predict_heart_disease,
    bg="#4CAF50",
    fg="white",
    font=button_font,
    relief="raised"
)
predict_button.grid(row=len(fields), column=0, columnspan=2, pady=20, ipadx=10, ipady=5)

# Load pre-trained models
best_bb_network  = load_model('best_nn_model.h5')  # Load Keras model
# best_bb_network = joblib.load('')  # Load BigML model
preprocessing_pipeline = joblib.load('preprocessing_pipeline.pkl')

# Start the GUI
root.mainloop()