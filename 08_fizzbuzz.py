# joelgrus/fizz-buzz-tensorflow
import numpy as np
import tensorflow as tf

import network.architectures


# One-hot encode the desired outputs: [number, "fizz", "buzz", "fizzbuzz"]
def fizz_buzz_encode(i):
    if   i % 15 == 0: return np.array([0, 0, 0, 1])
    elif i % 5  == 0: return np.array([0, 0, 1, 0])
    elif i % 3  == 0: return np.array([0, 1, 0, 0])
    else:             return np.array([1, 0, 0, 0])

# Our goal is to produce fizzbuzz for the numbers 1 to 100, training it on the
# numbers 101 to 1000
start, stop, split_point = 1, 1000, 100
X = np.arange(start, stop + 1, 1, np.float32)
y = np.array([fizz_buzz_encode(i) for i in X])

X_test, X_train = X[:split_point], X[split_point:]
y_test, y_train = y[:split_point], y[split_point:]


# TensorFlow needs to have a hashable type of input
X_train = X_train.reshape(X_train.size, 1)
X_test = X_test.reshape(X_test.size, 1)


# Try binary encoding
# Represent each input by an array of its binary digits.
# NUM_DIGITS = 10
#
#
# def binary_encode(i, num_digits):
#     return np.array([i >> d & 1 for d in range(num_digits)])
#
# X_train = np.array([binary_encode(int(i), NUM_DIGITS) for i in X_train],
#                    dtype=np.float32)
# X_test = np.array([binary_encode(int(i), NUM_DIGITS) for i in X_test],
#                   dtype=np.float32)


# We need a way to turn a prediction (and an original number)
# into a fizz buzz output
def fizz_buzz(i, prediction):
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]


# TensorFlow ##################################################################

# Device to use, either '/cpu:<x>' or '/gpu:<x>'
DEVICE = '/cpu:0'
# DEVICE = '/gpu:0'
TRAINING_EPOCHS = 5000

# Build the computational graph
# y_t = the true value of y
# y_p = the predicted value of y
x, y_t, y_p, w1, w2, b1, b2 = network.architectures.feed_forward_nn(
    device=DEVICE,
    # input_layer_size=NUM_DIGITS,
    output_layer_size=4,
    n_neurons=128,
    end_activation=None)

# Define the training step
with tf.device(device_name_or_function=DEVICE):
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_t, logits=y_p))
    train_step = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(y_p, 1), tf.argmax(y_t, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Launch the graph in a session
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# Train it
for e in range(TRAINING_EPOCHS):
    _, w1_val, w2_val, b1_val, b2_val = sess.run(
        (train_step, w1, w2, b1, b2), feed_dict={x: X_train, y_t: y_train})

    if e % 1000 == 0:
        train_error, train_accuracy = sess.run(
            (cost, accuracy), feed_dict={x: X_train, y_t: y_train})
        test_error, test_accuracy = sess.run(
            (cost, accuracy), feed_dict={x: X_test, y_t: y_test})

        print(
            ', '.join(["epoch {e}",
                       "train error {train_error}",
                       "train accuracy {train_accuracy}",
                       "test error {test_error}",
                       "test accuracy {test_accuracy}"]).format(
                e=e,
                train_error=train_error,
                train_accuracy=train_accuracy,
                test_error=test_error,
                test_accuracy=test_accuracy))

y_test_predicted = sess.run(y_p, feed_dict={x: X_test})

for i in range(split_point):
    print("N: {i}, Correct response: {c}, Predicted response: {p}".format(
        i=i+1,
        c=fizz_buzz(i, np.argmax(y_test_predicted[i])),
        p=fizz_buzz(i, np.argmax(y_test[i]))))
