# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from linear_regression import LinearRegressionModel
from tests import ModelEvaluation

# Load dataset
df = pd.read_csv("data/data.csv")

# Perform data cleaning
# DO I WANT ALL THE FEATURES?? -> convert sex to number

df_cleaned = df.dropna()

df_cleaned.loc[:, 'Sex'] = df_cleaned['Sex'].map({'male': 1, 'female': 0})

# Split into train and test sets

train_df, test_df = train_test_split(df_cleaned, test_size=0.2)

train_features = train_df.copy()
test_features = test_df.copy()

train_labels = train_features.pop("Survived")
test_labels = test_features.pop("Survived")
print(train_labels, test_labels)

# Initialize and train model


# make a tensor of the data 


# Evaluate model performance
