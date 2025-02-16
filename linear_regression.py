import tensorflow as tf
#### first commit ###

class LinearRegressionModel(tf.keras.Model):
    def __init__(self, input_dim):
        """
        Initializes a simple linear regression model.

        :param input_dim: Number of input features
        """
        super(LinearRegressionModel, self).__init__()
        # Define model layers
        self.dense = tf.keras.layers.Dense(1, input_dim=(input_dim), use_bias=True)

    def call(self, inputs):
        """
        Forward pass for the model.

        :param inputs: Input tensor
        :return: Predicted values
        """
        return self.dense(inputs)

    def train_model(self, X_train, y_train, epochs=100, lr=0.01):
        """
        Train the model using Mean Squared Error loss.

        :param X_train: Training features
        :param y_train: Training labels
        :param epochs: Number of training epochs
        :param lr: Learning rate
        """
        self.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                     loss='mse', metrics=['mae'])
        self.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)
