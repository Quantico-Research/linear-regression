# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from linear_regression import LinearRegressionModel
from tests import ModelEvaluation

# Load dataset
df = pd.read_csv("data/data.csv")

df = df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)


# Perform data cleaning

# Split into train and test sets

# Initialize and train model

# Evaluate model performance
