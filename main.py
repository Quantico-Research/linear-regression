# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from linear_regression import LinearRegressionModel
from tests import ModelEvaluation

# Load dataset
df = pd.read_csv("data/data.csv", quotechar='"')

# Drop columns not necessary for prediction
df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)

# Fill missing Age with median
df["Age"].fillna(df["Age"].median(), inplace=True)

# Fill missing Embarked with mode
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# Convert Sex to numeric (male = 1, female = 0)
df["Sex"] = df["Sex"].map({"male": 1, "female": 0})

# Convert columns to numeric types safely
numeric_columns = ['Pclass', 'Parch', 'Fare']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with NaN values in critical columns
df = df.dropna(subset=numeric_columns + ['Embarked'])

# Convert to integers where applicable
df["Pclass"] = df["Pclass"].astype(int)
df["Parch"] = df["Parch"].astype(int)

# One-hot encode Embarked
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

# Convert the one-hot encoded columns to integers
df["Embarked_Q"] = df["Embarked_Q"].astype(int)
df["Embarked_S"] = df["Embarked_S"].astype(int)

# Ensure Fare is float
df["Fare"] = df["Fare"].astype(float)

# Target and predictors
target = df["Survived"]
predictors = df.drop("Survived", axis=1)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    predictors, target, test_size=0.2, random_state=42
)

# Initialize and train the model
input_dim = X_train.shape[1]
model = LinearRegressionModel(input_dim=input_dim)

print("\n----Training Model----")
model.train_model(X_train, y_train, epochs=100, lr=0.01)

# Evaluate model performance
evaluation = ModelEvaluation(model, X_test, y_test)
mse = evaluation.evaluate_mse()
mae = evaluation.evaluate_mae()

print("\n--- Model Evaluation ---")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")

# Save predictions
predictions = evaluation.predict()
pd.DataFrame({"Predictions": predictions}).to_csv("predictions.csv", index=False)
print("\nPredictions saved to 'predictions.csv'")