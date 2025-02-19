import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_feature_importance(model, original_feature_names, preprocessing_pipeline):
    """
    Plots the feature importance of a trained model.

    Parameters:
        model: Trained model object with feature_importances_ attribute.
        original_feature_names: List of original feature names.
        preprocessing_pipeline: Preprocessing pipeline used to transform the features.
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("The model does not have a feature_importances_ attribute.")

    # Retrieve transformed feature names from the preprocessing pipeline
    if hasattr(preprocessing_pipeline, 'named_transformers_'):
        cat_encoder = preprocessing_pipeline.named_transformers_['cat']
        num_features = preprocessing_pipeline.named_transformers_['num'].feature_names_in_

        # Use feature_names_in_ directly from the encoder
        cat_features = cat_encoder.get_feature_names_out(input_features=cat_encoder.feature_names_in_)
        transformed_feature_names = list(num_features) + list(cat_features)
    else:
        raise ValueError("The preprocessing pipeline does not have the expected structure.")

    # Verify the lengths match
    if len(model.feature_importances_) != len(transformed_feature_names):
        raise ValueError("Mismatch between model feature importances and transformed feature names.")

    # Create a DataFrame for plotting
    feature_importances = pd.DataFrame({
        'Feature': transformed_feature_names,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importances, x='Importance', y='Feature')
    plt.title('Feature Importance')
    plt.show()


def plot_algorithm_performance(algorithms, scores):
    import pandas as pd
    import matplotlib.pyplot as plt

    # Ensure scores are scalar values
    results = pd.DataFrame({'Algorithm': algorithms, 'Accuracy': scores}).sort_values(by='Accuracy', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.bar(results['Algorithm'], results['Accuracy'], color='skyblue')
    plt.xlabel('Algorithm')
    plt.ylabel('Accuracy')
    plt.title('Algorithm Performance Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



def plot_correlation_heatmap(dataset):
    """
    Plots a correlation heatmap for the dataset.

    Parameters:
        dataset: DataFrame containing the dataset.
    """
    plt.figure(figsize=(12, 8))
    correlation_matrix = dataset.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
    plt.title("Correlation Heatmap")
    plt.show()

def plot_class_distribution(dataset, target_column):
    """
    Plots the distribution of the target variable.

    Parameters:
        dataset: DataFrame containing the dataset.
        target_column: Name of the target column.
    """
    class_counts = dataset[target_column].value_counts()
    plt.figure(figsize=(8, 6))
    sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()