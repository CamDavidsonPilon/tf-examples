import tensorflow as tf
import numpy as np

num_epochs = 1000

def model(X, w, b):
    return tf.add(tf.mul(X, w), b)


def inference(X):
    # X is a placeholder variable
    w = tf.Variable(tf.random_normal([1], stddev=0.01), name='weights')
    b = tf.Variable(tf.random_normal([1], stddev=0.01), name='biases')
    return model(X, w, b), w, b


def loss(y_hat, y):
    return tf.reduce_mean(tf.pow(y_hat - y, 2))


def train(loss):
    return tf.train.GradientDescentOptimizer(0.05).minimize(loss)


def predict(X, w, b):
    return model(X, w, b)


def run_training(x, y):
    X = tf.placeholder(tf.float32, x.shape)
    Y = tf.placeholder(tf.float32, y.shape)
    y_hat, w, b = inference(X)
    loss_op = loss(y_hat, Y)
    train_op = train(loss_op)
    predict_op = predict(X, w, b)


    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for step in xrange(num_epochs):
            _, loss_value, predictions = sess.run([train_op, loss_op, predict_op], feed_dict={X: x, Y: y})

            if step % 10 == 0:
                w_value, b_value = sess.run([w, b])
                print('Step %d: loss = %.2f: w: %.2f: b: %.2f' % (step, loss_value, w_value, b_value))


if __name__ == '__main__':    

    w = 4.
    b = 2.

    x = np.random.randn(100)
    y = w*x + b + np.random.randn(100)

    run_training(x, y)
