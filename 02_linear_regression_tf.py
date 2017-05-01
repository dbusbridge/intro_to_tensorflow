import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

import responses
import network.architectures

# Global configuration
BIAS = 2
GRADIENT = 3
X_RANGE_MIN, X_RANGE_MAX = -100, 100
N = 1000
TEST_SIZE = 0.33

# Create the data
X = np.random.uniform(low=X_RANGE_MIN, high=X_RANGE_MAX, size=N)
y = responses.linear(X, bias=BIAS, gradient=GRADIENT, noise_sd=10)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)

# Plot training data ##########################################################
plt.scatter(X_train, y_train,  color='black')
plt.xlabel("X")
plt.ylabel("y")


# TensorFlow needs to have a hashable type of input
X_train = X_train.reshape(X_train.size, 1)
X_test = X_test.reshape(X_test.size, 1)
y_train = y_train.reshape(y_train.size, 1)
y_test = y_test.reshape(y_test.size, 1)


# TensorFlow ##################################################################

# Device to use, either '/cpu:<x>' or '/gpu:<x>'
DEVICE = '/cpu:0'
# DEVICE = '/gpu:0'
TRAINING_EPOCHS = 50000

# Start the session
sess = tf.InteractiveSession()

# Build the computational graph
# y_t = the true value of y
# y_p = the predicted value of y
x, y_t, y_p, w, b = network.architectures.linear_regression(device=DEVICE)

# Define the training step
with tf.device(device_name_or_function=DEVICE):
    with tf.name_scope('cost'):
        cost = tf.reduce_mean(tf.square(y_p - y_t))

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('logs/train', sess.graph)
test_writer = tf.summary.FileWriter('logs/test')

# Initialise the variables
sess.run(tf.global_variables_initializer())

# Train me :D
fd_train = {x: X_train, y_t: y_train}
fd_test = {x: X_test, y_t: y_test}

# (Comment on error versus noise)
for e in range(TRAINING_EPOCHS):
    _, w_val, b_val = sess.run((train_step, w, b), feed_dict=fd_train)

    if e % 10 == 0:
        summary_train = sess.run(merged, feed_dict=fd_train)
        train_writer.add_summary(summary_train, e)

        summary_test = sess.run(merged, feed_dict=fd_test)
        test_writer.add_summary(summary_test, e)

    if e % 100 == 0:
        train_error, test_error = cost.eval(
            feed_dict=fd_train), cost.eval(feed_dict=fd_test)

        print(
            ', '.join(["epoch {e}",
                       "train error {train_error}",
                       "test error {test_error}",
                       "w {w_val}",
                       "b {b_val}"]).format(
                e=e,
                train_error=train_error,
                test_error=test_error,
                w_val=w_val[0],
                b_val=b_val[0]))

train_writer.close()
test_writer.close()

# Plot results ################################################################
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, y_p.eval(feed_dict={x: X_test}), color='red', linewidth=3)
plt.xlabel("X")
plt.ylabel("y")

# plt.show()
# tensorboard --logdir=logs