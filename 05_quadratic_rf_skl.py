import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import responses

# Global configuration
BIAS = 2
GRADIENT = 3
X_RANGE_MIN, X_RANGE_MAX = -1, 1
N = 1000
TEST_SIZE = 0.33

# Create the data
X = np.random.uniform(low=X_RANGE_MIN, high=X_RANGE_MAX, size=N)
y = responses.quadratic(X, bias=BIAS, gradient=GRADIENT, noise_sd=0.1)

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
regr_rf = RandomForestRegressor(max_depth=10)

# Train the model using the training sets
regr_rf.fit(X_train, y_train)

# Plot results ################################################################
plt.scatter(X_test, y_test,  color='black')
plt.plot(np.sort(X_test, axis=0),
         regr_rf.predict(np.sort(X_test, axis=0)),
         color='blue', linewidth=3)
plt.xlabel("X")
plt.ylabel("y")

# plt.show()
