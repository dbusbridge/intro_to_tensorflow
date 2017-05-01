import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split

import responses

# Global configuration
BIAS = 2
GRADIENT = 3
X_RANGE_MIN, X_RANGE_MAX = -100, 100
N = 1000
TEST_SIZE = 0.33

# Create the data
X = np.random.uniform(low=X_RANGE_MIN, high=X_RANGE_MAX, size=N)
y = responses.linear(X, bias=BIAS, gradient=GRADIENT, noise_sd=10)

# Reshape X to be a 2-dimensional array
X = X.reshape(X.size, 1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)

# Plot training data ##########################################################
plt.scatter(X_train, y_train,  color='black')
plt.xlabel("X")
plt.ylabel("y")


# SciKit-Learn ################################################################

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# The coefficients
print('Coefficients: \n', regr.coef_)

# Plot results ################################################################
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, regr.predict(X_test), color='blue', linewidth=3)
plt.xlabel("X")
plt.ylabel("y")

# plt.show()
