
---

## 📌 **README.md for Cardiovascular Deep Learning Project**

```
# 🏥 Cardiovascular Disease Prediction using Machine Learning & Deep Learning

This project is a **comprehensive Cardiovascular Disease Prediction System** that utilizes **Machine Learning (ML) and Deep Learning (DL) models** to predict heart disease risk based on patient medical attributes.

It includes **multiple ML models, a deep learning neural network, and a GUI application** to provide **real-time predictions**.

---

## 🔗 **Repository Link**
📌 GitHub: [Cardiovascular Deep Learning](https://github.com/meshtirop1/cardiovascular_deep_learning-.git)

## 📧 **Contact**
📩 Email: [mtirop345@gmail.com](mailto:mtirop345@gmail.com)

---

## 📖 **Table of Contents**
- [📝 Project Overview](#-project-overview)
- [✨ Features](#-features)
- [📦 Installation & Setup](#-installation--setup)
- [📊 Dataset](#-dataset)
- [🛠️ Model Training & Performance](#️-model-training--performance)
- [🖥️ GUI Application](#️-gui-application)
- [📈 Results & Visualization](#-results--visualization)
- [📂 File Structure](#-file-structure)
- [💡 Future Enhancements](#-future-enhancements)
- [🤝 Acknowledgments](#-acknowledgments)

---

## 📝 **Project Overview**

This system processes **cardiovascular health data**, applies **data preprocessing & feature engineering**, trains multiple **Machine Learning & Deep Learning models**, and provides **real-time predictions via a GUI**.

The models implemented include:

✔ **Logistic Regression**  
✔ **XGBoost**  
✔ **Naive Bayes**  
✔ **Support Vector Machine (SVM)**  
✔ **Artificial Neural Network (ANN)**  

The **Graphical User Interface (GUI)** is built using **Tkinter**, allowing users to input patient data and get real-time predictions.

---

## ✨ **Features**

✅ **Data Loading & Exploration** (`data_loader.py`)  
✅ **Preprocessing & Feature Engineering** (`preprocessing.py`)  
✅ **Multiple Machine Learning Model Implementations** (`models.py`)  
✅ **Deep Learning Neural Network (ANN)** (`neural_network.py`)  
✅ **Graphical User Interface (GUI) for Predictions** (`frontend.py`)  
✅ **Performance Visualization & Analysis** (`visualizations.py`)  

---

## 📦 **Installation & Setup**

### 🔹 **Step 1: Clone the Repository**
```sh
git clone https://github.com/meshtirop1/cardiovascular_deep_learning-.git
cd cardiovascular_deep_learning-
```

### 🔹 **Step 2: Install Dependencies**
Ensure you have Python installed (`>=3.7`), then run:
```sh
pip install -r requirements.txt
```

### 🔹 **Step 3: Prepare the Dataset**
Place the dataset (`heart.csv`) in the root directory.

---

## 📊 **Dataset**

The dataset consists of **303 patient records**, with 13 input features and 1 target variable (`target: 1 = Disease present, 0 = No disease`).

#### 📌 **Dataset Information**
```
Total Records: 303
Features: 13 (Age, Cholesterol, Blood Pressure, etc.)
Target: Binary (1 = Heart Disease, 0 = No Disease)
```

#### 📊 **Feature List:**
| Feature Name       | Description |
|--------------------|-------------|
| **age**           | Patient age (years) |
| **sex**           | 1 = Male, 0 = Female |
| **cp**           | Chest pain type (0-3) |
| **trestbps**      | Resting blood pressure |
| **chol**         | Serum cholesterol (mg/dl) |
| **fbs**          | Fasting blood sugar (1=Yes, 0=No) |
| **restecg**       | Resting ECG results (0-2) |
| **thalach**      | Maximum heart rate achieved |
| **exang**        | Exercise-induced angina (1=Yes, 0=No) |
| **oldpeak**      | ST depression induced by exercise |
| **slope**        | Slope of peak exercise ST segment (0-2) |
| **ca**          | Number of major vessels (0-3) |
| **thal**         | Thalassemia (1-3) |

---

## 🛠️ **Model Training & Performance**

### 🔹 **Step 1: Train the Machine Learning Models**
Run the main script to:
- Load & preprocess data
- Train various ML models
- Evaluate performance

```sh
python main.py
```

📊 **Model Performance:**
```
Logistic Regression Accuracy: 88.52%
XGBoost Accuracy: 83.61%
Rainforest Accuracy: 85.61%
Naive Bayes Accuracy: 85.25%
Support Vector Machine Accuracy: 81.97%
```

### 🔹 **Step 2: Train the Neural Network**
To train the deep learning model:

```sh
python neural_network.py
```

📌 **Best Neural Network Model Saved To**: `best_nn_model.h5`  
📊 **Best Validation Accuracy:** `91.80%`  

---

## 🖥️ **GUI Application**

The GUI, built with **Tkinter**, allows real-time predictions based on user inputs.

### 🔹 **Run the GUI**
```sh
python frontend.py
```

### 🔹 **Usage**
- Enter patient data
- Click the **Predict** button
- View results (Positive or Negative for heart disease)

---

## 📈 **Results & Visualization**

Performance metrics and visualization tools are available in `visualizations.py`, including:

✔ **Feature Importance Plots**  
✔ **Algorithm Comparison Charts**  
✔ **Data Distribution & Correlation Heatmaps**  

### 🔹 **To generate visualizations:**
```sh
python visualizations.py
```

---

## 📂 **File Structure**
```
cardiovascular_deep_learning-/
│── data_loader.py          # Load & explore dataset
│── preprocessing.py        # Data preprocessing
│── models.py               # Machine Learning models
│── neural_network.py       # Deep Learning model (ANN)
│── visualizations.py       # Visualization functions
│── frontend.py             # GUI for predictions
│── main.py                 # Main script (trains models)
│── requirements.txt        # Dependencies list
│── README.md               # Documentation
│── heart.csv               # Dataset (ensure it's added)
```

---

## 💡 **Future Enhancements**
🔹 **Deploy model using Flask or FastAPI**  
🔹 **Optimize Neural Network** for better accuracy  
🔹 **Use a larger dataset** to improve generalization  
🔹 **Implement an API for remote model inference**  

---

## 🤝 **Acknowledgments**
- **Kaggle Heart Disease Dataset**  
- **Scikit-Learn, TensorFlow/Keras, XGBoost, and Tkinter**  
- Developed by **[meshtirop1](https://github.com/meshtirop1/)**  

---

## 🎯 **Final Notes**
- **Ensure `heart.csv` is in the project directory** before running the scripts.
- **The trained models will be saved automatically for future use.**
- **GUI makes real-time predictions based on user inputs.**

🚀 **Happy Coding!**  
```

---

### 📌 **Next Steps**
1️⃣ **Save this as `README.md` in your project root**  
2️⃣ **Commit & Push to GitHub**
```sh
git add .
git commit -m "Updated README with detailed model performance"
git push origin main
```

