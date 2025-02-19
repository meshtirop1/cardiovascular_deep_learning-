from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import joblib

def logistic_regression(X_train, X_test, Y_train, Y_test):
    lr = LogisticRegression()
    lr.fit(X_train, Y_train)
    Y_pred = lr.predict(X_test)
    return evaluate_model("Logistic Regression", Y_test, Y_pred)

# def random_forest(X_train, X_test, Y_train, Y_test, save_path="random_forest_model.pkl"):
#     rf = RandomForestClassifier()
#     param_grid = {
#         'n_estimators': [100, 300, 500],
#         'max_depth': [10, 20, 30, None],
#         'min_samples_split': [2, 5, 10],
#         'min_samples_leaf': [1, 2, 4],
#         'max_features': ['sqrt', 'log2', None],
#         #'bootstrap': [True, False]
#     }
#
#     grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='accuracy', cv=5, verbose=2, n_jobs=-1)
#     grid_search.fit(X_train, Y_train)
#     best_rf = grid_search.best_estimator_
#     joblib.dump(best_rf, save_path)  # Save model
#     print(f"Random Forest model saved to {save_path}.")
#     Y_pred = best_rf.predict(X_test)
#     return evaluate_model("Random Forest", Y_test, Y_pred), best_rf

def xgboost_model(X_train, X_test, Y_train, Y_test, save_path="xgboost_model.pkl"):
    xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
    xgb_model.fit(X_train, Y_train)
    Y_pred = xgb_model.predict(X_test)
    joblib.dump(xgb_model, save_path)  # Save model
    print(f"XGBoost model saved to {save_path}.")
    return evaluate_model("XGBoost", Y_test, Y_pred)

def naive_bayes(X_train, X_test, Y_train, Y_test, save_path="naive_bayes_model.pkl"):
    nb = GaussianNB()
    nb.fit(X_train, Y_train)
    Y_pred = nb.predict(X_test)
    joblib.dump(nb, save_path)  # Save model
    print(f"Naive Bayes model saved to {save_path}.")
    return evaluate_model("Naive Bayes", Y_test, Y_pred)

def support_vector_machine(X_train, X_test, Y_train, Y_test, save_path="svm_model.pkl"):
    svm = SVC(kernel='linear', probability=True)
    svm.fit(X_train, Y_train)
    Y_pred = svm.predict(X_test)
    joblib.dump(svm, save_path)  # Save model
    print(f"SVM model saved to {save_path}.")
    return evaluate_model("Support Vector Machine", Y_test, Y_pred)

def evaluate_model(name, Y_test, Y_pred):
    accuracy = round(accuracy_score(Y_test, Y_pred) * 100, 2)
    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    print(f"{name} Accuracy: {accuracy}%")
    print(f"Precision: {precision}, Recall: {recall}, F1-Score: {f1}")
    return accuracy, precision, recall, f1

#
# from sklearn.model_selection import train_test_split
# import numpy as np
#
#
# def ensemble_blending(X_train, X_test, Y_train, Y_test):
#     # Split train set into training and validation sets
#     X_blend_train, X_blend_valid, Y_blend_train, Y_blend_valid = train_test_split(X_train, Y_train, test_size=0.2,
#                                                                                   random_state=42)
#
#     # Train base models
#     model1 = LogisticRegression(max_iter=1000, random_state=42)
#     model2 = RandomForestClassifier(n_estimators=200, random_state=42)
#     model3 = SVC(probability=True, random_state=42)
#
#     model1.fit(X_blend_train, Y_blend_train)
#     model2.fit(X_blend_train, Y_blend_train)
#     model3.fit(X_blend_train, Y_blend_train)
#
#     # Generate predictions for the validation set
#     blend_features = np.column_stack([
#         model1.predict_proba(X_blend_valid)[:, 1],
#         model2.predict_proba(X_blend_valid)[:, 1],
#         model3.predict_proba(X_blend_valid)[:, 1]
#     ])
#
#     # Train meta-model on blended features
#     meta_model = LogisticRegression(random_state=42)
#     meta_model.fit(blend_features, Y_blend_valid)
#
#     # Generate predictions for the test set
#     test_features = np.column_stack([
#         model1.predict_proba(X_test)[:, 1],
#         model2.predict_proba(X_test)[:, 1],
#         model3.predict_proba(X_test)[:, 1]
#     ])
#     Y_pred = meta_model.predict(test_features)
#
#     return evaluate_model("Blending Classifier", Y_test, Y_pred)
#
#
# from sklearn.ensemble import StackingClassifier
# from sklearn.tree import DecisionTreeClassifier
#
#
# def ensemble_stacking(X_train, X_test, Y_train, Y_test):
#     # Define base models
#     base_models = [
#         ('lr', LogisticRegression(max_iter=1000, random_state=42)),
#         ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
#         ('svc', SVC(probability=True, random_state=42))
#     ]
#
#     # Define meta-model
#     meta_model = DecisionTreeClassifier(max_depth=5, random_state=42)
#
#     # Create stacking classifier
#     stacking_clf = StackingClassifier(
#         estimators=base_models,
#         final_estimator=meta_model,
#         cv=5
#     )
#
#     # Train the stacking classifier
#     stacking_clf.fit(X_train, Y_train)
#
#     # Make predictions
#     Y_pred = stacking_clf.predict(X_test)
#
#     return evaluate_model("Stacking Classifier", Y_test, Y_pred)
#
#
# from sklearn.ensemble import VotingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
#
#
# def ensemble_voting(X_train, X_test, Y_train, Y_test):
#     # Define individual models
#     model1 = LogisticRegression(max_iter=1000, random_state=42)
#     model2 = RandomForestClassifier(n_estimators=200, random_state=42)
#     model3 = SVC(probability=True, random_state=42)
#
#     # Create a voting classifier
#     voting_clf = VotingClassifier(
#         estimators=[
#             ('lr', model1),
#             ('rf', model2),
#             ('svc', model3)
#         ],
#         voting='soft'  # Use 'soft' for probability averaging
#     )
#
#     # Train the voting classifier
#     voting_clf.fit(X_train, Y_train)
#
#     # Make predictions
#     Y_pred = voting_clf.predict(X_test)
#
#     return evaluate_model("Voting Classifier", Y_test, Y_pred)
