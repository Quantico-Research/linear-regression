import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error

class ModelEvaluation:
    def __init__(self, model, X_test, y_test):
        """
        Initializes model evaluation.

        :param model: Trained model
        :param X_test: Test features
        :param y_test: Test labels
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def predict(self):
        """
        Generates predictions using the trained model.
        """
        # returns results as a 1-D array
        return self.model.predict(self.X_test).flatten() 

    def evaluate_mse(self):
        """
        Computes Mean Squared Error (MSE).
        """
        y_pred = self.predict()
        return mean_squared_error(self.y_test, y_pred)

    def evaluate_mae(self):
        """
        Computes Mean Absolute Error (MAE).
        """
        y_pred = self.predict()
        return mean_absolute_error(self.y_test, y_pred)
