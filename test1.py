# coding=utf-8
# user=hu

import tensorflow as tf
import numpy as np


def t1():
    # create date
    x_date = np.random.rand(100).astype(np.float32)
    y_date = x_date*0.1 + 0.3

    Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    biases = tf.Variable(tf.zeros([1]))

    y = Weights*x_date + biases

    loss = tf.reduce_mean(tf.square(y - y_date))

    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    for step in range(20000):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(Weights), sess.run(biases))


def t3():
    # 定义变量
    state = tf.Variable(0, name='counter')
    # 定义常量
    one = tf.constant(1)
    # 定义加法步骤（注意：此步并没有直接计算
    new_value = tf.add(state, one)
    update = tf.assign(state, new_value)
    # 定义了Variable一定要initialize
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for _ in range(3):
            sess.run(update)
            print(sess.run(state))

def placeholder_test():
    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)

    output = tf.multiply(input1, input2)

    with tf.Session() as sess:
        print(sess.run(output, feed_dict={input1: [7.], input2: [9.]}))


def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)

    return outputs

if __name__ == '__main__':
    placeholder_test()