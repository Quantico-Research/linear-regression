import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error,
    accuracy_score,
    precision_score,
    f1_score,
    confusion_matrix
)

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
        predictions = self.model.predict(self.X_test)
        return predictions

    def evaluate_mse(self):
        """
        Computes Mean Squared Error (MSE).
        """
        predictions = self.predict()
        mse = mean_squared_error(self.y_test, predictions)
        return mse

    def evaluate_mae(self):
        """
        Computes Mean Absolute Error (MAE).
        """
        predictions = self.predict()
        mae = mean_absolute_error(self.y_test, predictions)
        return mae

    def evaluate_accuracy(self):
        """
        Computes classification accuracy.
        """
        predictions = self.predict()
        # Convert probabilities to binary predictions
        binary_predictions = (predictions > 0.5).astype(int)
        accuracy = accuracy_score(self.y_test, binary_predictions)
        return accuracy
    
    def evaluate_precision(self):
        """
        Computes classification accuracy.
        """
        predictions = self.predict()
        # Convert probabilities to binary predictions
        binary_predictions = (predictions > 0.5).astype(int)
        precision = precision_score(self.y_test, binary_predictions)
        return precision
        
    def evaluate_f1(self):
        """
        Computes F1 score.
        """
        predictions = self.predict()
        binary_predictions = (predictions > 0.5).astype(int)
        f1 = f1_score(self.y_test, binary_predictions)
        return f1
        
    def plot_confusion_matrix(self, save_path=None):
        """
        Plots confusion matrix for better visualization.
        
        :param save_path: Path to save the confusion matrix image
        """
        predictions = self.predict()
        binary_predictions = (predictions > 0.5).astype(int)
        
        cm = confusion_matrix(self.y_test, binary_predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path)
            print(f"Confusion matrix saved to {save_path}")
            
        plt.close()