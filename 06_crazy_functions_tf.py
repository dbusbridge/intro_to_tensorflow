import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

import responses
import network.architectures

# Global configuration
X_RANGE_MIN, X_RANGE_MAX = -2 * np.pi, 2 * np.pi
N = 1000
TEST_SIZE = 0.33

# Create the data
X = np.random.uniform(low=X_RANGE_MIN, high=X_RANGE_MAX, size=N)
y = responses.general(X, f=lambda a: a * np.cos(a), noise_sd=0.1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)

# TensorFlow needs to have a hashable type of input
X_train = X_train.reshape(X_train.size, 1)
X_test = X_test.reshape(X_test.size, 1)
y_train = y_train.reshape(y_train.size, 1)
y_test = y_test.reshape(y_test.size, 1)


# TensorFlow ##################################################################

# Device to use, either '/cpu:<x>' or '/gpu:<x>'
# DEVICE = '/cpu:0'
DEVICE = '/gpu:0'
TRAINING_EPOCHS = 25000

# Start the session
sess = tf.InteractiveSession()

# Build the computational graph
# y_t = the true value of y
# y_p = the predicted value of y
x, y_t, y_p, w1, w2, b1, b2 = network.architectures.feed_forward_nn(
    device=DEVICE)

# Define the training step
with tf.device(device_name_or_function=DEVICE):
    cost = tf.reduce_mean(tf.square(y_p - y_t))
    train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

# Initialise the variables
sess.run(tf.global_variables_initializer())

# Train me :D
# (Comment on error versus noise)
for e in range(TRAINING_EPOCHS):
    _, w1_val, w2_val, b1_val, b2_val = sess.run(
        (train_step, w1, w2, b1, b2), feed_dict={x: X_train, y_t: y_train})

    if e % 1000 == 0:
        train_error = cost.eval(feed_dict={x: X_train, y_t: y_train})
        test_error = cost.eval(feed_dict={x: X_test, y_t: y_test})
        print(
            ', '.join(["epoch {e}",
                       "train error {train_error}",
                       "test error {test_error}"]).format(
                e=e,
                train_error=train_error,
                test_error=test_error))

# Plot results ################################################################
plt.scatter(X_test, y_test,  color='black')
plt.plot(np.sort(X_test, axis=0),
         y_p.eval(feed_dict={x: np.sort(X_test, axis=0)}),
         color='red', linewidth=3)
plt.xlabel("X")
plt.ylabel("y")

# plt.show()
