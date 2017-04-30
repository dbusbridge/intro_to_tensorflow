import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
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












# Try something else ##########################################################

# Feature engineering
X2_train, X2_test = X_train ** 2, X_test ** 2

# SciKit-Learn (round two baby!) ##############################################

# Create linear regression object
regr2 = linear_model.LinearRegression()

# Train the model using the training sets
regr2.fit(X2_train, y_train)

# The coefficients
print('Coefficients: \n', regr2.coef_)

# Plot results ################################################################
plt.scatter(X2_test, y_test,  color='black')
plt.plot(X2_test, regr2.predict(X2_test), color='blue', linewidth=3)
plt.xlabel("X2")
plt.ylabel("y")

# plt.show()

# Problem with this is that now separate points in the X-feature space map to
# the same location in the X2 feature space. What if there are subtle
# differences?
