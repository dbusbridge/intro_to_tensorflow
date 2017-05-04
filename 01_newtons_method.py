import os

import tensorflow as tf

# 
# http://danielhomola.com/2016/02/09/newtons-method-with-10-lines-of-python/
def dx(f, x):
    return abs(f(x))


def newtons_method(f, df, x0, e):
    delta = dx(f, x0)
    while delta > e:
        x0 = x0 - f(x0) / df(x0)
        delta = dx(f, x0)
    print('Root is at: ', x0)
    print('f(x) at root is: ', f(x0))


def f(x):
    return 6 * x ** 5 - 5 * x ** 4 - 4 * x ** 3 + 3 * x ** 2


def df(x):
    return 30 * x ** 4 - 20 * x ** 3 - 12 * x ** 2 + 6 * x


x0s = [0., .5, 1.]
for x0 in x0s:
    newtons_method(f, df, x0, 1e-5)


###############################################################################
# Sort of newtons-method-with-almost-10-lines-of-python-with-TensorFlow
# def newtons_method_tf(f, x0, e):
#     x0_tf = tf.Variable(initial_value=[x0], dtype=tf.float32)
#     loss = tf.abs(f(x0_tf))
#     train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
#     sess = tf.InteractiveSession()
#     sess.run(tf.global_variables_initializer())
#     delta = sess.run(loss, feed_dict={})
#     while delta > e:
#         delta, _ = sess.run((loss, train_step), feed_dict={})
#     print('Root is at: ', sess.run(x0_tf, feed_dict={}))
#     print('f(x) at root is: ', f(sess.run(x0_tf, feed_dict={})))
#
#
# x0s = [0., .5, 1.]
# for x0 in x0s:
#     newtons_method_tf(f, x0, 1e-5)


# Sort of newtons-method-with-almost-10-lines-of-python-with-TensorFlow-with-
# TensorBoard
def newtons_method_tf_tb(f, x0, e):
    log_path = os.path.join('logs/newton', str(x0))

    with tf.name_scope('root'):
        x0_tf = tf.Variable(initial_value=x0, dtype=tf.float32, name="x0")
        tf.summary.scalar('x0', x0_tf)

    with tf.name_scope('delta'):
        loss = tf.abs(f(x0_tf), name="loss")
        tf.summary.scalar('loss', loss)

    with tf.name_scope('training'):
        train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

    summ = tf.summary.merge_all()

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter(log_path)
    writer.add_graph(sess.graph)

    i = 0
    delta, s = sess.run((loss, summ), feed_dict={})
    writer.add_summary(s, i)
    while delta > e:
        delta, _, s = sess.run((loss, train_step, summ), feed_dict={})
        i += 1
        writer.add_summary(s, i)

    print('Root is at: ', sess.run(x0_tf, feed_dict={}))
    print('f(x) at root is: ', f(sess.run(x0_tf, feed_dict={})))


x0s = [0.4]
for x0 in x0s:
    newtons_method_tf_tb(f, x0, 1e-5)
