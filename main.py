# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from linear_regression import LinearRegressionModel
from tests import ModelEvaluation

# Load dataset
df = pd.read_csv("data/data.csv")

# Perform data cleaning
df = df.drop(columns="Survived")
df_cleaned = df.dropna()

# Split into train and test sets

# Initialize and train model

# Evaluate model performance
