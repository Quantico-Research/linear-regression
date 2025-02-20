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
        self.dense1 = tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,))
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.output_layer = tf.keras.layers.Dense(1) 

    def call(self, inputs, training=False):
        """
        Forward pass for the model.

        :param inputs: Input tensor
        :return: Predicted values
        """

        x = self.dense1(inputs)
        x = self.batch_norm1(x, training=training)
        x = self.dense2(x)
        x = self.batch_norm2(x, training=training)
        return self.output_layer(x)

    def train_model(self, X_train, y_train, epochs=100, lr=0.01):
        """
        Train the model using Mean Squared Error loss.

        :param X_train: Training features
        :param y_train: Training labels
        :param epochs: Number of training epochs
        :param lr: Learning rate
        """

        self.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                    loss='mse', 
                    metrics=['mae'])
        
        # Add callbacks for better training
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001
            )
        ]
        
        return self.fit(X_train, y_train, 
                       epochs=epochs,
                       batch_size=32,
                       validation_split=0.1,
                       callbacks=callbacks)