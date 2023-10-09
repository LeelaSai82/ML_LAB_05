#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#A5


# In[1]:


import numpy as np

# Customer data
customer_data = [
    [20, 6, 2, 386, 1],  # Yes
    [16, 3, 6, 289, 1],  # Yes
    [27, 6, 2, 393, 1],  # Yes
    [19, 1, 2, 110, 0],  # No
    [24, 4, 2, 280, 1],  # Yes
    [22, 1, 5, 167, 0],  # No
    [15, 4, 2, 271, 1],  # Yes
    [18, 4, 2, 274, 1],  # Yes
    [21, 1, 4, 148, 0],  # No
    [16, 2, 4, 198, 0],  # No
]

# Convert the data to NumPy arrays
data = np.array(customer_data)

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Initialize weights and bias
np.random.seed(0)
weights = np.random.randn(4)
bias = np.random.randn(1)

# Set the learning rate
learning_rate = 0.01

# Training loop
epochs = 10000
for epoch in range(epochs):
    for row in data:
        inputs = row[:4]
        label = row[4]

        # Forward propagation
        z = np.dot(weights, inputs) + bias
        predicted = sigmoid(z)

        # Calculate the error
        error = label - predicted

        # Update weights and bias using gradient descent
        weights += learning_rate * error * predicted * (1 - predicted) * inputs
        bias += learning_rate * error * predicted * (1 - predicted)

# Make predictions on the data
predictions = []
for row in data:
    inputs = row[:4]
    z = np.dot(weights, inputs) + bias
    predicted = sigmoid(z)
    if predicted >= 0.5:
        predictions.append("Yes")
    else:
        predictions.append("No")

# Print the predictions
for i, prediction in enumerate(predictions):
    print(f"C_{i+1}: Predicted - {prediction}")


# In[ ]:


#A1


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#loading the project data
df = pd.read_excel(r"C:\Users\leela\Downloads\embeddingsdata (1).xlsx")
df


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# Define the initial weights and learning rate
W0 = 10
W1 = 0.2
W2 = -0.75
learning_rate = 0.05

# Load your data from a DataFrame (binary_df)
# Assuming 'embed_1' and 'embed_2' are your input features, and 'Label' is your target label.
binary_df = df[df['Label'].isin([0, 1])]
X = binary_df[['embed_1', 'embed_2']]
y = binary_df['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize variables for tracking epochs and errors
epochs = 0
errors = []

# Training the perceptron
max_epochs = 1000  # Set a maximum number of epochs to avoid infinite looping

while epochs < max_epochs:
    total_error = 0

    for i in range(0, len(X_train)):
        # Calculate the weighted sum of inputs
        weighted_sum = W0 + W1 * X_train.iloc[i]['embed_1'] + W2 * X_train.iloc[i]['embed_2']

        # Apply step activation function element-wise
        prediction = 1 if weighted_sum >= 0 else 0

        # Calculate the error
        error = y_train.iloc[i] - prediction

        # Update weights
        W0 = W0 + learning_rate * error
        W1 = W1 + learning_rate * error * X_train.iloc[i]['embed_1']
        W2 = W2 + learning_rate * error * X_train.iloc[i]['embed_2']

        total_error += error ** 2

    # Append the total error for this epoch to the list
    errors.append(total_error)

    # Check for convergence
    if total_error == 0:
        break

    epochs += 1

# Plot epochs against error values
plt.plot(range(epochs), errors)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Epochs vs. Error in Perceptron Training')
plt.show()

# Test the trained perceptron on the test data
correct_predictions = 0

for i in range(len(X_test)):
    weighted_sum = W0 + W1 * X_test.iloc[i]['embed_1'] + W2 * X_test.iloc[i]['embed_2']

    # Apply step activation function element-wise
    prediction = 1 if weighted_sum >= 0 else 0

    if prediction == y_test.iloc[i]:
        correct_predictions += 1

accuracy = correct_predictions / len(X_test)

print(f"Accuracy on Test Data: {accuracy * 100:.2f}%")
print(f"Final Weights: W0 = {W0}, W1 = {W1}, W2 = {W2}")
print(f"Number of Epochs: {epochs}")


# In[ ]:


#A2


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# Define the initial weights and learning rate
W0 = 10
W1 = 0.2
W2 = -0.75
learning_rate = 0.05

# Load your data from a DataFrame (binary_df)
# Assuming 'embed_1' and 'embed_2' are your input features, and 'Label' is your target label.
binary_df = df[df['Label'].isin([0, 1])]
X = binary_df[['embed_1', 'embed_2']]
y = binary_df['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize variables for tracking epochs and errors
epochs = 0
errors = []

# Define activation functions
def bipolar_step(x):
    return -1 if x < 0 else 1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return max(0, x)

# Select the activation function to use
activation_function = sigmoid  # Change this to bipolar_step or relu for different activations

# Training the perceptron
max_epochs = 1000  # Set a maximum number of epochs to avoid infinite looping

while epochs < max_epochs:
    total_error = 0

    for i in range(0, len(X_train)):
        # Calculate the weighted sum of inputs
        weighted_sum = W0 + W1 * X_train.iloc[i]['embed_1'] + W2 * X_train.iloc[i]['embed_2']

        # Apply the selected activation function
        prediction = activation_function(weighted_sum)

        # Calculate the error
        error = y_train.iloc[i] - prediction

        # Update weights
        W0 = W0 + learning_rate * error
        W1 = W1 + learning_rate * error * X_train.iloc[i]['embed_1']
        W2 = W2 + learning_rate * error * X_train.iloc[i]['embed_2']

        total_error += error ** 2

    # Append the total error for this epoch to the list
    errors.append(total_error)

    # Check for convergence
    if total_error == 0:
        break

    epochs += 1

# Plot epochs against error values
plt.plot(range(epochs), errors)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Epochs vs. Error in Perceptron Training')
plt.show()

# Test the trained perceptron on the test data
correct_predictions = 0

for i in range(len(X_test)):
    weighted_sum = W0 + W1 * X_test.iloc[i]['embed_1'] + W2 * X_test.iloc[i]['embed_2']

    # Apply the selected activation function
    prediction = activation_function(weighted_sum)

    if prediction == y_test.iloc[i]:
        correct_predictions += 1

accuracy = correct_predictions / len(X_test)

print(f"Activation Function: {activation_function.__name__}")
print(f"Accuracy on Test Data: {accuracy * 100:.2f}%")
print(f"Final Weights: W0 = {W0}, W1 = {W1}, W2 = {W2}")
print(f"Number of Epochs: {epochs}")


# In[ ]:


#A3


# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# Define the initial weights
W0 = 10
W1 = 0.2
W2 = -0.75

# Load your data from a DataFrame (binary_df)
# Assuming 'embed_1' and 'embed_2' are your input features, and 'Label' is your target label.
binary_df = df[df['Label'].isin([0, 1])]
X = binary_df[['embed_1', 'embed_2']]
y = binary_df['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize variables for tracking epochs and errors for each learning rate
learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
iterations_to_converge = []

for learning_rate in learning_rates:
    W0_current = W0
    W1_current = W1
    W2_current = W2
    
    # Initialize variables for tracking epochs and errors
    epochs = 0
    errors = []

    # Training the perceptron
    max_epochs = 1000  # Set a maximum number of epochs to avoid infinite looping

    while epochs < max_epochs:
        total_error = 0

        for i in range(0, len(X_train)):
            # Calculate the weighted sum of inputs
            weighted_sum = W0_current + W1_current * X_train.iloc[i]['embed_1'] + W2_current * X_train.iloc[i]['embed_2']

            # Apply step activation function element-wise
            prediction = 1 if weighted_sum >= 0 else 0

            # Calculate the error
            error = y_train.iloc[i] - prediction

            # Update weights
            W0_current = W0_current + learning_rate * error
            W1_current = W1_current + learning_rate * error * X_train.iloc[i]['embed_1']
            W2_current = W2_current + learning_rate * error * X_train.iloc[i]['embed_2']

            total_error += error ** 2

        # Check for convergence
        if total_error == 0:
            break

        epochs += 1
    
    iterations_to_converge.append(epochs)

# Plot learning rates against the number of iterations taken to converge
plt.plot(learning_rates, iterations_to_converge, marker='o', linestyle='-')
plt.xlabel('Learning Rate')
plt.ylabel('Iterations to Converge')
plt.title('Learning Rate vs. Iterations to Converge')
plt.grid(True)
plt.show()


# In[ ]:

import numpy as np
import matplotlib.pyplot as plt

# Define the initial weights and learning rate
W0 = -1  # Adjusted for XOR gate
W1 = 1
W2 = 1
learning_rate = 0.1

# XOR gate training data (inputs and corresponding targets)
data = np.array([[0, 0, 0],
                 [0, 1, 1],
                 [1, 0, 1],
                 [1, 1, 0]])

# Step activation function
def step(x):
    return 1 if x >= 0 else 0

# Training loop
epochs = 10000
error_values = []

for epoch in range(epochs):
    total_error = 0

    for row in data:
        inputs = row[:2]
        target = row[2]

        # Calculate the weighted sum
        weighted_sum = W0 + W1 * inputs[0] + W2 * inputs[1]

        # Apply step activation function
        prediction = step(weighted_sum)

        # Calculate the error
        error = target - prediction

        # Update weights and bias using the perceptron learning rule
        W0 = W0 + learning_rate * error
        W1 = W1 + learning_rate * error * inputs[0]
        W2 = W2 + learning_rate * error * inputs[1]

        total_error += error ** 2

    # Calculate the mean squared error for the epoch
    mean_squared_error = total_error / len(data)
    error_values.append(mean_squared_error)

    # Check for convergence (error is zero)
    if mean_squared_error == 0:
        break

# Plot epochs vs. error values
plt.figure()
plt.plot(range(len(error_values)), error_values, marker='o', linestyle='-')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.title('Training Progress')
plt.grid(True)
plt.show()

# Print the final weights and bias
print("Final weights:")
print(f"W0: {W0}")
print(f"W1: {W1}")
print(f"W2: {W2}")

#A4
#A4
import numpy as np

# Define initial weights and learning rate
W0 = 10
W1 = 0.2
W2 = -0.75
learning_rate = 0.05


# Training data for XOR gate
# XOR gate truth table: inputs and corresponding outputs
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([0, 1, 1, 0])

def activate(sum):
    return 1 if sum >= 0 else 0

# Perceptron training function
def train_perceptron(weights, learning_rate, max_epochs, data):
    errors = []  # To store error values for each epoch
    for epoch in range(max_epochs):
        total_error = 0
        for i in range(len(data)):
            x1, x2 = data[i]
            target = targets[i]
            # Calculate the weighted sum
            weighted_sum = weights[0] + weights[1] * x1 + weights[2] * x2
            # Calculate the error
            error = target - activate(weighted_sum)
            total_error += error
            # Update weights
            weights[0] += learning_rate * error
            weights[1] += learning_rate * error * x1
            weights[2] += learning_rate * error * x2
        errors.append(total_error)
        if total_error == 0:
            break
    return weights, errors

# Train the perceptron and collect errors
trained_weights, error_values = train_perceptron([W0, W1, W2], learning_rate, 100, inputs)

# Print the trained weights
print("Trained Weights:")
print(f"W0: {trained_weights[0]}, W1: {trained_weights[1]}, W2: {trained_weights[2]}")

# Test the perceptron
def test_perceptron(weights, data):
    correct = 0
    for i in range(len(data)):
        x1, x2 = data[i]
        target = targets[i]
        weighted_sum = weights[0] + weights[1] * x1 + weights[2] * x2
        prediction = activate(weighted_sum)
        if prediction == target:
            correct += 1
        print(f"Input: ({x1}, {x2}), Target: {target}, Prediction: {prediction}")
    accuracy = (correct / len(data)) * 100
    print(f"Accuracy: {accuracy}%")

# Test the trained perceptron
print("\nTesting the Trained Perceptron:")
test_perceptron(trained_weights, inputs)

#A6
# Calculate the pseudo-inverse of the data
pseudo_inverse = np.linalg.pinv(df)
print("Pseudo inverse is",pseudo_inverse)

#A7
#A7
import numpy as np

class ANDGateNeuralNetwork:
    def __init__(self, learning_rate=0.05):
        self.weights_ih = np.random.randn(2, 2)
        self.weights_ho = np.random.randn(1, 2)
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward_propagate(self, inputs):
        h = self.sigmoid(np.dot(self.weights_ih, inputs))
        o = self.sigmoid(np.dot(self.weights_ho, h))
        return o

    def backpropagate(self, inputs, target_output, actual_output, h):
        error = target_output - actual_output

        gradient_ho = error * actual_output * (1 - actual_output)
        gradient_ih = (gradient_ho @ self.weights_ho) * h * (1 - h)

        self.weights_ho += self.learning_rate * np.outer(gradient_ho, h)
        self.weights_ih += self.learning_rate * np.outer(gradient_ih, inputs)

    def train(self, training_examples):
        for inputs, target_output in training_examples:
            h = self.sigmoid(np.dot(self.weights_ih, inputs))
            actual_output = self.forward_propagate(inputs)
            self.backpropagate(inputs, target_output, actual_output, h)

    def predict(self, inputs):
        return self.forward_propagate(inputs)

# Create a new AND gate neural network
network = ANDGateNeuralNetwork()

# Train the network on the AND gate truth table
training_examples = [(np.array([0, 0]), 0), (np.array([0, 1]), 0), (np.array([1, 0]), 0), (np.array([1, 1]), 1)]
network.train(training_examples)

# Test the network for multiple inputs
inputs_list = [np.array([0, 0]), np.array([0, 1]), np.array([1, 0]), np.array([1, 1])]

for inputs in inputs_list:
    output = network.predict(inputs)
    print(f"Input: {inputs}, Output: {output}")

#A8

#A8
import numpy as np

class XORGateNeuralNetwork:
    def __init__(self, learning_rate=0.05):
        self.weights_ih = np.random.randn(2, 2)
        self.weights_ho = np.random.randn(1, 2)
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward_propagate(self, inputs):
        h = self.sigmoid(np.dot(self.weights_ih, inputs))
        o = self.sigmoid(np.dot(self.weights_ho, h))
        return o

    def backpropagate(self, inputs, target_output, actual_output, h):
        error = target_output - actual_output

        gradient_ho = error * actual_output * (1 - actual_output)
        gradient_ih = (gradient_ho @ self.weights_ho) * h * (1 - h)

        self.weights_ho += self.learning_rate * np.outer(gradient_ho, h)
        self.weights_ih += self.learning_rate * np.outer(gradient_ih, inputs)

    def train(self, training_examples):
        for inputs, target_output in training_examples:
            h = self.sigmoid(np.dot(self.weights_ih, inputs))
            actual_output = self.forward_propagate(inputs)
            self.backpropagate(inputs, target_output, actual_output, h)

    def predict(self, inputs):
        return self.forward_propagate(inputs)

#Create a new XOR gate neural network
network = XORGateNeuralNetwork()

#Train the network on the XOR gate truth table
training_examples = [(np.array([0, 0]), 0), (np.array([0, 1]), 1), (np.array([1, 0]), 1), (np.array([1, 1]), 0)]
network.train(training_examples)

#Test the network for multiple inputs
inputs_list = [np.array([0, 0]), np.array([0, 1]), np.array([1, 0]), np.array([1, 1])]

for inputs in inputs_list:
    output = network.predict(inputs)
    print(f"Input: {inputs}, Output: {output}")

#A9
#A9
import numpy as np

class Perceptron:
    def __init__(self, num_inputs, learning_rate=0.05):
        self.weights = np.random.randn(num_inputs)
        self.learning_rate = learning_rate

    def forward_propagate(self, inputs):
        weighted_sum = np.dot(inputs, self.weights)
        output = 1 / (1 + np.exp(-weighted_sum))  # Use sigmoid activation
        return output

    def backpropagate(self, inputs, target_output, actual_output):
        error = target_output - actual_output
        delta = error * actual_output * (1 - actual_output)
        self.weights += self.learning_rate * delta * inputs

    def train(self, training_examples, num_epochs=100):
        for epoch in range(num_epochs):
            for inputs, target_output in training_examples:
                actual_output = self.forward_propagate(inputs)
                self.backpropagate(inputs, target_output, actual_output)

    def predict(self, inputs):
        return self.forward_propagate(inputs)

# Create a new perceptron with 2 input features (for AND and XOR)
num_inputs = 2
perceptron = Perceptron(num_inputs)

# Create a training dataset for AND
training_and = [
    (np.array([0, 0]), 0),
    (np.array([0, 1]), 0),
    (np.array([1, 0]), 0),
    (np.array([1, 1]), 1)
]

# Create a training dataset for XOR
training_xor = [
    (np.array([0, 0]), 0),
    (np.array([0, 1]), 1),
    (np.array([1, 0]), 1),
    (np.array([1, 1]), 0)
]

# Train the perceptron for AND
perceptron.train(training_and)

# Test the perceptron for AND
inputs_and = np.array([1, 1])
output_and = perceptron.predict(inputs_and)
print("AND Gate:", output_and)

# Train the perceptron for XOR
perceptron.train(training_xor)

# Test the perceptron for XOR
inputs_xor = np.array([1, 1])
output_xor = perceptron.predict(inputs_xor)
print("XOR Gate:", output_xor)

for inputs in inputs_list:
    output = network.predict(inputs)
    print(f"Input: {inputs}, Output: {output}")

#A10
#A10
import numpy as np
from sklearn.neural_network import MLPClassifier

# Training data for AND gate
# AND gate truth table: inputs and corresponding outputs
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

# Create an MLPClassifier with one hidden layer
mlp = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic', solver='sgd', learning_rate_init=0.05, max_iter=100)

# Train the classifier
mlp.fit(X, y)

# Print the trained weights and biases
print("Trained Weights (Coefs):")
print(mlp.coefs_)
print("Trained Biases (Intercepts):")
print(mlp.intercepts_)

# Test the trained classifier
def test_classifier(classifier, data, targets):
    predictions = classifier.predict(data)
    accuracy = (sum(predictions == targets) / len(targets)) * 100
    print("Predictions:", predictions)
    print("Accuracy:", accuracy, "%")

# Test the trained classifier
print("\nTesting the Trained Classifier:")
test_classifier(mlp, X, y)

#A11
import numpy as np
import pandas as pd  # You need to import pandas for DataFrame operations
from sklearn.model_selection import train_test_split

# Define the initial weights and learning rate
W0 = 10
W1 = 0.2
W2 = -0.75
learning_rate = 0.05

# Assuming you have loaded 'df' and 'activation_maps' previously
binary_df = df[df['Label'].isin([0, 1])]
X = binary_df[['embed_1', 'embed_2']]
y = binary_df['Label']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize variables for tracking epochs and errors
epochs = 0
errors = []

# Define the Step activation function
def step_activation(x):
    return 1 if x >= 0 else 0

# Training the perceptron
max_epochs = 1000  # Set a maximum number of epochs to avoid infinite looping

while epochs < max_epochs:
    total_error = 0
    
    for i in range(len(X_train)):
        # Calculate the weighted sum of inputs
        weighted_sum = W0 + W1 * X_train.iloc[i, 0] + W2 * X_train.iloc[i, 1]
        
        # Apply the Step activation function
        prediction = step_activation(weighted_sum)
        
        # Calculate the error
        error = y_train.iloc[i] - prediction
        
        # Update weights
        W0 = W0 + learning_rate * error
        W1 = W1 + learning_rate * error * X_train.iloc[i, 0]
        W2 = W2 + learning_rate * error * X_train.iloc[i, 1]
        
        total_error += error ** 2
    
    # Append the total error for this epoch to the list
    errors.append(total_error)
    
    # Check for convergence
    if total_error == 0:
        break
    
    epochs += 1

# Test the trained perceptron
for i in range(len(X_test)):
    weighted_sum = W0 + W1 * X_test.iloc[i, 0] + W2 * X_test.iloc[i, 1]
    prediction = step_activation(weighted_sum)
    print(f"Input: {X_test.iloc[i].values}, Target: {y_test.iloc[i]}, Predicted: {prediction}")

print(f"Final Weights: W0 = {W0}, W1 = {W1}, W2 = {W2}")
print(f"Number of Epochs: {epochs}")



