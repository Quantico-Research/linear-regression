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

df = pd.read_csv("data/data.csv")

# Drop columns not necessary to prediction
df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis = 1, inplace=True)

# Fill missing Age with median
df["Age"].fillna(df["Age"].median(), inplace=True)

# Fill missing Embarked with mode
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# Convert Sex to numeirc (male = 1, female = 0)
df["Sex"] = df["Sex"].map({"male": 1, "female": 0})

# Convert the "Embarked" column into multiple binary (0/1) columns (one-hot encoding),
# while dropping the first category to avoid redundancy.
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

# Target column
target = df["Survived"]

# Predictive columns
predictors = df.drop(["Survived"], axis = 1)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
        predictors, target, test_size=0.2, random_state=42
)






# Split into train and test sets

# Initialize and train model

# Evaluate model performance
