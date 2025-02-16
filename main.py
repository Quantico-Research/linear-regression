# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from linear_regression import LinearRegressionModel
from tests import ModelEvaluation

# Load dataset
df = pd.read_csv("data/data.csv")

# Perform data cleaning
df = df.dropna()
df = df.drop(columns=["Name", "Ticket", "Cabin"], inplace=True)

# Split into train and test sets
X = df.drop(columns="Survived")
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

train_df, test_df = train_test_split(df_cleaned, test_size=0.2)


# Initialize and train model
input_dim = X_train.shape[1]
model = LinearRegressionModel(input_dim)
model.fit(X_train, y_train, epochs=100, lr=0.01)


# Evaluate model performance
