# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from linear_regression import LinearRegressionModel
from tests import ModelEvaluation
import numpy as np
from sklearn.preprocessing import StandardScaler


# Load dataset
df = pd.read_csv("data/data.csv")

# Perform data cleaning
# DO I WANT ALL THE FEATURES?? -> convert sex to number

df_cleaned = df.dropna()
df_cleaned = df_cleaned.drop(columns=["PassengerId", "Name","Ticket", "Cabin", "Embarked"])


df_cleaned.loc[:, 'Sex'] = df_cleaned['Sex'].map({'male': 1, 'female': 0})

df_cleaned = df_cleaned.astype('float32')

# Split into train and test sets
train_df, test_df = train_test_split(df_cleaned, test_size=0.2)

train_features = train_df.copy()
test_features = test_df.copy()

# get prediction variable
train_labels = train_features.pop("Survived")
test_labels = test_features.pop("Survived")

# Scale features to 0 and 1
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)


# Initialize and train model
model = LinearRegressionModel(6)

model.train_model(train_features, train_labels, epochs=200, lr=0.1)


# Evaluate model performance
evaluator = ModelEvaluation(model, test_features, test_labels)
prediction = evaluator.predict()
mae = evaluator.evaluate_mae()
mse = evaluator.evaluate_mse()
print("prediction:", prediction)
print("mae:", mae)
print("mse:", mse)


baseline_prediction = np.mean(train_labels)
baseline_mae = np.mean(np.abs(test_labels - baseline_prediction))
print("Baseline MAE:", baseline_mae)