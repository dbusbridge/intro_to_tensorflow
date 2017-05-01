import tensorflow as tf


def weight(shape, stddev=0.1):
    """
    Create weights used in neural networks by sampling from a truncated
    normal distribution.
    :param list[int] shape: The dimensions of the weight variable to create.
    :param dbl stddev: The standard deviation of the normal distribution to
        sample the weights from. Defaults to 0.1.
    :return: A weight matrix.
    :rtype: tf.Tensor.
    """
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)


def bias(shape, value=0.1):
    """
    Create the bias used in neural networks as a constant shift.
    :param list[int] shape: The dimensions of the bias variable to create.
    :param dbl value: The bias to create. Defaults to 0.1.
    :return: A bias matrix.
    :rtype: tf.Tensor.
    """
    initial = tf.constant(value=value, shape=shape)
    return tf.Variable(initial)


def make_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
