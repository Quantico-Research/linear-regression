import tensorflow as tf

class LinearRegressionModel(tf.keras.Model):
    def __init__(self, input_dim):
        """
        Initializes a simple linear regression model.

        :param input_dim: Number of input features
        """
        # Define model layers
        super(LinearRegressionModel, self).__init__()
        self.x = tf.keras.Input(shape= (input_dim,))
        self.y = tf.keras.Dense(1, activation = None)
        

    def call(self, inputs):
        """
        Forward pass for the model.

        :param inputs: Input tensor
        :return: Predicted values
        """
        vars  = self.x(inputs)
        return self.y(vars)

        pass  # Implement forward pass

    def train_model(self, X_train, y_train, epochs=100, lr=0.01):
        """
        Train the model using Mean Squared Error loss.

        :param X_train: Training features
        :param y_train: Training labels
        :param epochs: Number of training epochs
        :param lr: Learning rate
        """
        # model.compile
        # model.fit
        pass  # Implement training loop
