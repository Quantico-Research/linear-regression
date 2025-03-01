# Import necessary libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import tensorflow as tf
from linear_regression import LinearRegressionModel
from tests import ModelEvaluation

def main():
    """
    Main function to run the entire pipeline.
    """
    print("Starting Titanic Survival Prediction Pipeline...")
    
    # Create output directories
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv("data/data.csv")
    
    # Basic data exploration
    print(f"Dataset shape: {df.shape}")
    print("\nSample data:")
    print(df.head())
    
    # Visualize survival distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Survived', data=df)
    plt.title('Survival Distribution')
    plt.tight_layout()
    plt.savefig('results/survival_distribution.png')
    plt.close()
    
    # Preprocess data
    print("\nPreprocessing data...")
    
    # Drop irrelevant columns
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
    
    # Convert data types
    df['Parch'] = pd.to_numeric(df['Parch'], errors='coerce')
    df['Fare'] = pd.to_numeric(df['Fare'], errors='coerce')
    df['Pclass'] = df['Pclass'].astype(str)
    
    # Define feature types
    numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
    categorical_features = ['Sex', 'Embarked', 'Pclass']
    
    # Create preprocessing pipelines
    numeric_transformer = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler()
    )
    
    categorical_transformer = make_pipeline(
        SimpleImputer(strategy='constant', fill_value='missing'),
        OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    )
    
    # Combine numeric and categorical pipelines
    preprocessor = make_column_transformer(
        (numeric_transformer, numeric_features),
        (categorical_transformer, categorical_features)
    )

    # Convert values 2 and 3, treating them as "Survived" (1)
    df['Survived'] = df['Survived'].apply(lambda x: 1 if x > 0 else 0)
    
    # Split into features and target
    X = df.drop(columns="Survived", axis=1)
    y = df["Survived"]
    
    # Split into train and test sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Preprocess data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Initialize and train model
    input_dim = X_train_processed.shape[1]
    model = LinearRegressionModel(input_dim)
    model.train_model(X_train_processed, y_train, epochs=100, lr=0.01)
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluation = ModelEvaluation(model, X_test_processed, y_test)
    
    # Calculate metrics
    mse = evaluation.evaluate_mse()
    mae = evaluation.evaluate_mae()
    accuracy = evaluation.evaluate_accuracy()
    precision = evaluation.evaluate_precision()
    f1 = evaluation.evaluate_f1()
    
    # Plot confusion matrix
    evaluation.plot_confusion_matrix(save_path='results/confusion_matrix.png')
    
    # Print results
    print("\nResults:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Create a results summary figure
    plt.figure(figsize=(10, 6))
    metrics = ['MSE', 'MAE', 'Accuracy', 'Precision', 'F1 Score']
    values = [mse, mae, accuracy, precision, f1]

    # Plot the metrics
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'] 
    bars = plt.bar(metrics, values, color=colors)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')

    plt.ylim(0, 1.0)  # Set y-axis limit to 0-1 for better visualization
    plt.title('Model Performance Metrics')
    plt.tight_layout()
    plt.savefig('results/metrics_summary.png')
    plt.close()
    
    print("\nPipeline completed successfully!")
    print("All visualizations saved to the 'results' directory")

if __name__ == "__main__":
    main()
