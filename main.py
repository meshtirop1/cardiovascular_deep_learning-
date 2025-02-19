from data_loader import load_data, explore_data
from preprocessing import preprocess_data
from models import (
    logistic_regression, xgboost_model, naive_bayes, support_vector_machine
)
from neural_network import build_and_train_nn
import joblib

# Load and explore data
try:
    dataset = load_data("heart.csv")
except FileNotFoundError:
    print("Error: Dataset file 'heart.csv' not found.")
    exit(1)

explore_data(dataset)

# Preprocess data
X_train_scaled, X_test_scaled, Y_train, Y_test, preprocessing_pipeline = preprocess_data(dataset)
joblib.dump(preprocessing_pipeline, 'preprocessing_pipeline.pkl')

# Train and evaluate models
print("\nTraining Logistic Regression...")
lr_acc, _, _, _ = logistic_regression(X_train_scaled, X_test_scaled, Y_train, Y_test)

print("\nTraining XGBoost...")
xgb_acc, _, _, _ = xgboost_model(X_train_scaled, X_test_scaled, Y_train, Y_test)

print("\nTraining Naive Bayes...")
nb_acc, _, _, _ = naive_bayes(X_train_scaled, X_test_scaled, Y_train, Y_test)

print("\nTraining Support Vector Machine...")
svm_acc, _, _, _ = support_vector_machine(X_train_scaled, X_test_scaled, Y_train, Y_test)

# Neural Network
print("\nTraining Neural Network...")
nn_model, history = build_and_train_nn(X_train_scaled, Y_train, X_test_scaled, Y_test)
# Commented out for now: plot_training_history(history)

# Compare performances
algorithms = [
    "Logistic Regression",
    "XGBoost",
    "Naive Bayes",
    "Support Vector Machine",
]
scores = [lr_acc, xgb_acc, nb_acc, svm_acc]
# Uncomment to visualize: plot_algorithm_performance(algorithms, scores)
