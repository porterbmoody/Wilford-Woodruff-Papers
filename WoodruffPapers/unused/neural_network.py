#%%
import numpy as np

class NeuralNetwork:
    def __init__(self, learning_rate):
        self.weights = np.array([np.random.randn(), np.random.randn()])
        self.bias = np.random.randn()
        self.learning_rate = learning_rate

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_deriv(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def predict(self, input_vector):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2
        return prediction

    def _compute_gradients(self, input_vector, target):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2

        derror_dprediction = 2 * (prediction - target)
        dprediction_dlayer1 = self._sigmoid_deriv(layer_1)
        dlayer1_dbias = 1
        dlayer1_dweights = (0 * self.weights) + (1 * input_vector)

        derror_dbias = (
            derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
        )
        derror_dweights = (
            derror_dprediction * dprediction_dlayer1 * dlayer1_dweights
        )

        return derror_dbias, derror_dweights

    def _update_parameters(self, derror_dbias, derror_dweights):
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - (
            derror_dweights * self.learning_rate
        )


#%%

learning_rate = 0.1
input_vector = np.array([2, 1.5])


input_vector
#%%
neural_network = NeuralNetwork(learning_rate)

neural_network.predict(input_vector)
neural_network.weights

#%%

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Generate random training data
num_samples = 100
input_dim = 10
num_classes = 3

# Generate random input features (x_train)
x_train = np.random.rand(num_samples, input_dim)

# Generate random labels (y_train)
y_train = np.random.randint(num_classes, size=num_samples)
y_train = keras.utils.to_categorical(y_train, num_classes)

# Generate random testing data
num_test_samples = 200

# Generate random input features (x_test)
x_test = np.random.rand(num_test_samples, input_dim)

# Generate random labels (y_test)
y_test = np.random.randint(num_classes, size=num_test_samples)
y_test = keras.utils.to_categorical(y_test, num_classes)


y_test
#%5
# Define the architecture of the ANN
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, batch_size=32, epochs=10, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)






