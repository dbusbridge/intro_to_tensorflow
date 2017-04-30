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
