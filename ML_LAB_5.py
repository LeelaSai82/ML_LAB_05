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




