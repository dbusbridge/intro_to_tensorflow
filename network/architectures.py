import tensorflow as tf
import network.variables as variables


def linear_regression(device='/cpu:0'):
    """Create a neural network that has an architecture that matches linear
    regression.
    :param str device: The device to use for storing variables and computation.
        Either '/cpu:<n>' or '/cpu:<n>'. Defaults to '/cpu:<n>.
    :return: A set of TensorFlow tensors that serve as inputs, controllers and
        outputs of the network.
    :rtype: (tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor).
    """
    with tf.device(device_name_or_function=device):
        # Placeholders for feed and output
        x = tf.placeholder(tf.float32, shape=[None, 1])
        y_true = tf.placeholder(tf.float32, shape=[None, 1])

        # First layer: Weights + bias
        w = variables.weight(shape=[1])
        b = variables.bias(shape=[1])

        # Readout layer: perform linear transformation + bias with relu
        y_pred = tf.multiply(x, w) + b

    return x, y_true, y_pred, w, b


def feed_forward_nn(device='/cpu:0'):
    """Create a neural network that has one hidden layer.
    :param str device: The device to use for storing variables and computation.
        Either '/cpu:<n>' or '/cpu:<n>'. Defaults to '/cpu:<n>.
    :return: A set of TensorFlow tensors that serve as inputs, controllers and
        outputs of the network.
    :rtype: (tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor).
    """
    with tf.device(device_name_or_function=device):
        # Placeholders for feed and output
        x = tf.placeholder(tf.float32, shape=[None, 1])
        y_true = tf.placeholder(tf.float32, shape=[None, 1])

        # First layer: Weights + bias
        w1 = variables.weight(shape=[1, 50])
        b1 = variables.bias(shape=[50])

        # Readout layer: perform linear transformation + bias with relu
        h1 = tf.nn.relu(tf.multiply(x, w1) + b1)

        # Second layer: Weights + bias
        w2 = variables.weight(shape=[50, 1])
        b2 = variables.bias(shape=[1])

        y_pred = tf.matmul(h1, w2) + b2

    return x, y_true, y_pred, w1, w2, b1, b2
