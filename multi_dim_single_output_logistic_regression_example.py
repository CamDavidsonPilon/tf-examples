import tensorflow as tf
import numpy as np
from utils import SEED

np.random.seed(SEED)

def sigmoid(x):
    return 1./(1 + np.exp(-x))


def model(X, w, b):
    return tf.matmul(X, w) + b


def inference(X):
    # X is a placeholder variable
    w = tf.Variable(tf.random_normal([X.get_shape()[1].value, 1], stddev=0.1, seed=SEED), name='weights')
    b = tf.Variable(tf.random_normal([1], stddev=0.1, seed=SEED), name='biases')
    return model(X, w, b), w, b


def loss(y_hat, y):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_hat, y))


def train(loss):
    return tf.train.GradientDescentOptimizer(0.05).minimize(loss)


def predict(X, w, b):
    return tf.sigmoid(model(X, w, b))


def run_training(x, y):
    num_epochs = 4000
    X = tf.placeholder(tf.float32, shape=x.shape)
    Y = tf.placeholder(tf.float32, shape=y.shape)
    y_hat, w, b = inference(X)
    loss_op = loss(y_hat, Y)
    train_op = train(loss_op)
    predict_op = predict(X, w, b)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for step in xrange(num_epochs):
            _, loss_value, p = sess.run([train_op, loss_op, predict_op], feed_dict={X: x, Y: y})

            if step % 100 == 0:
                w_value, b_value = sess.run([w, b])
                print('Step %d: loss = %.2f' % (step, loss_value))
                #print (y * -np.log(p) + (1 - y) * -np.log(1 - p)).mean()
                #mprint w_value, b_value

    return w_value, b_value


if __name__ == '__main__':


    N = 1000
    w = np.random.randn(1,10)
    b = 0.1

    x = np.random.randn(N, w.shape[1])
    p = sigmoid(x[:, 0])[:, None]*sigmoid(np.dot(x, w.T) + b)
    y = np.random.binomial(1, p)
    run_training(x, y)