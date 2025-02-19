import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def load_data(filepath):
    """
    Loads the dataset from the given file path.

    Parameters:
        filepath (str): Path to the dataset file.

    Returns:
        DataFrame: Loaded dataset as a pandas DataFrame.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        dataset = pd.read_csv(filepath)
        print(f"Dataset loaded successfully with {dataset.shape[0]} rows and {dataset.shape[1]} columns.")
        return dataset
    except Exception as e:
        raise ValueError(f"Error loading dataset: {e}")

def explore_data(dataset):
    """
    Provides basic exploratory visualizations for the dataset.

    Parameters:
        dataset (DataFrame): The dataset to explore.
    """
    print("Basic Dataset Information:\n")
    print(dataset.info())
    print("\nDescriptive Statistics:\n")
    print(dataset.describe())

    for column in dataset.select_dtypes(include=['float64', 'int64']).columns:
        sns.histplot(dataset[column], kde=True, bins=20)
        plt.title(f"Distribution of {column}")
        plt.show()

    missing_values = dataset.isnull().sum()
    print("Missing Values per Column:")
    print(missing_values[missing_values > 0])