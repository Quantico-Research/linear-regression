import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from linear_regression import LinearRegressionModel
from tests import ModelEvaluation

# Load dataset
df = pd.read_csv("data/data.csv")

# Drop irrelevant columns
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

# Convert data types
df['Parch'] = pd.to_numeric(df['Parch'], errors='coerce')
df['Fare'] = pd.to_numeric(df['Fare'], errors='coerce')
df['Pclass'] = df['Pclass'].astype(str)

# Define feature types
numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
categorical_features = ['Sex', 'Embarked', 'Pclass']

# Create preprocessing pipelines for both numeric and categorical data
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

# Split into train and test sets
X = df.drop(columns="Survived", axis=1)
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Preprocess data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Initialize and train model
input_dim = X_train_processed.shape[1]
model = LinearRegressionModel(input_dim)
model.train_model(X_train_processed, y_train, epochs=100, lr=0.01)

# Evaluate model performance
evaluation = ModelEvaluation(model, X_test_processed, y_test)
mse = evaluation.evaluate_mse()
mae = evaluation.evaluate_mae()

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")