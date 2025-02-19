# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# import pandas as pd
# import joblib
# import numpy as np
#
#
# def preprocess_data(dataset, test_size=0.2, random_state=0, save_pipeline_path="preprocessing_pipeline.pkl"):
#     """
#     Preprocess the dataset by scaling numerical features and encoding categorical features.
#
#     Parameters:
#         dataset: DataFrame containing the dataset.
#         test_size: Proportion of data to include in the test split.
#         random_state: Seed for reproducibility.
#         save_pipeline_path (str): Path to save the preprocessing pipeline.
#
#     Returns:
#         Scaled and encoded training and testing data, and preprocessing pipeline.
#     """
#     if not isinstance(dataset, pd.DataFrame):
#         raise TypeError("Input dataset must be a pandas DataFrame.")
#
#     if 'target' not in dataset.columns:
#         raise KeyError("The dataset does not contain the required 'target' column.")
#
#     print("Initial Dataset Info:")
#     print(dataset.info())
#     print("\nInitial Descriptive Statistics:")
#     print(dataset.describe())
#
#     if dataset.isnull().any().any():
#         print("WARNING: Dataset contains missing values. Imputing with mean.")
#         dataset = dataset.fillna(dataset.mean())
#
#     print("\nDataset Info after missing value handling:")
#     print(dataset.info())
#     print("\nDescriptive Statistics after missing value handling:")
#     print(dataset.describe())
#
#     X = dataset.drop("target", axis=1)
#     y = dataset["target"]
#
#     print("\nFeature Data Types before preprocessing:")
#     print(X.dtypes)
#
#     categorical_features = ['sex', 'cp', 'restecg', 'slope', 'ca', 'thal']
#     numerical_features = [col for col in X.columns if col not in categorical_features]
#
#     print("\nNumerical Features:", numerical_features)
#     print("Categorical Features:", categorical_features)
#
#     preprocess = ColumnTransformer([
#         ('scale_numeric', StandardScaler(), numerical_features),
#         ('encode_categorical', OneHotEncoder(handle_unknown='ignore'), categorical_features)
#     ])
#
#     X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
#
#     print("\nShape of X_train before preprocessing:", X_train.shape)
#     print("Shape of X_test before preprocessing:", X_test.shape)
#
#     X_train_scaled = preprocess.fit_transform(X_train)
#     X_test_scaled = preprocess.transform(X_test)
#
#     print("\nShape of X_train after preprocessing:", X_train_scaled.shape)
#     print("Shape of X_test after preprocessing:", X_test_scaled.shape)
#
#     # Check for NaN in scaled data
#     if np.isnan(X_train_scaled).any():
#         raise ValueError("NaN values found in scaled training data.")
#     if np.isnan(X_test_scaled).any():
#         raise ValueError("NaN values found in scaled testing data.")
#
#     # Save the preprocessing pipeline
#     joblib.dump(preprocess, save_pipeline_path)
#     print(f"\nPreprocessing pipeline saved to {save_pipeline_path}.")
#
#     return X_train_scaled, X_test_scaled, Y_train, Y_test, preprocess
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import joblib
import numpy as np
from collections import Counter


def custom_smote(X, y, random_state=0):
    """
    Custom implementation of SMOTE for oversampling the minority class.

    Parameters:
        X: Feature matrix.
        y: Target labels.
        random_state: Seed for reproducibility.

    Returns:
        Oversampled feature matrix and target labels.
    """
    np.random.seed(random_state)
    unique_classes = np.unique(y)
    class_counts = Counter(y)

    max_count = max(class_counts.values())
    X_resampled, y_resampled = list(X), list(y)

    for cls in unique_classes:
        count = class_counts[cls]
        if count < max_count:
            # Find all samples from the current class
            class_indices = np.where(y == cls)[0]
            delta = max_count - count
            synthetic_samples = []

            # Generate synthetic samples
            for _ in range(delta):
                idx1, idx2 = np.random.choice(class_indices, 2, replace=False)
                sample = X[idx1] + np.random.random() * (X[idx2] - X[idx1])
                synthetic_samples.append(sample)

            X_resampled.extend(synthetic_samples)
            y_resampled.extend([cls] * delta)

    return np.array(X_resampled), np.array(y_resampled)


def preprocess_data(dataset, test_size=0.2, random_state=0, save_pipeline_path="preprocessing_pipeline.pkl",
                    apply_smote=False):
    """
    Preprocess the dataset by scaling numerical features, encoding categorical features,
    and optionally applying custom SMOTE for imbalanced datasets.

    Parameters:
        dataset: DataFrame containing the dataset.
        test_size: Proportion of data to include in the test split.
        random_state: Seed for reproducibility.
        save_pipeline_path (str): Path to save the preprocessing pipeline.
        apply_smote (bool): Whether to apply custom SMOTE for oversampling minority class.

    Returns:
        Scaled and encoded training and testing data, and preprocessing pipeline.
    """
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError("Input dataset must be a pandas DataFrame.")

    if 'target' not in dataset.columns:
        raise KeyError("The dataset does not contain the required 'target' column.")

    if dataset.isnull().any().any():
        print("WARNING: Dataset contains missing values. Imputing with mean.")
        dataset = dataset.fillna(dataset.mean())

    # Separate features and target
    X = dataset.drop("target", axis=1).values
    y = dataset["target"].values

    categorical_features = ['sex', 'cp', 'restecg', 'slope', 'ca', 'thal']
    numerical_features = [col for col in dataset.columns if col not in categorical_features + ['target']]

    preprocess = ColumnTransformer([
        ('scale_numeric', StandardScaler(), numerical_features),
        ('encode_categorical', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Apply the preprocessing pipeline
    X_train_preprocessed = preprocess.fit_transform(pd.DataFrame(X_train, columns=dataset.columns[:-1]))
    X_test_preprocessed = preprocess.transform(pd.DataFrame(X_test, columns=dataset.columns[:-1]))

    if apply_smote:
        print("\nApplying custom SMOTE for oversampling minority class...")
        X_train_preprocessed, Y_train = custom_smote(X_train_preprocessed, Y_train, random_state=random_state)

    # Save the preprocessing pipeline
    joblib.dump(preprocess, save_pipeline_path)
    print(f"\nPreprocessing pipeline saved to {save_pipeline_path}.")

    return X_train_preprocessed, X_test_preprocessed, Y_train, Y_test, preprocess
