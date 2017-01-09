# lets try linear regression
import tensorflow as tf
import numpy as np
from sklearn.cross_validation import train_test_split


class Data(object):

    def __init__(self, x, y):
        # hold back 20% for testing
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)

        self.xtr = x_train
        self.ytr = y_train

        self.Xtr = tf.placeholder(tf.float32, shape=x_train.shape, name='xtrain')
        self.Ytr = tf.placeholder(tf.float32, shape=y_train.shape, name='ytrain')

        self.xte = x_test
        self.Xte = tf.placeholder(tf.float32, shape=x_test.shape, name='xtest')


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, 1)[:, None])
    return e_x / e_x.sum(1)[:, None]


def model(X, w, b):
    return tf.matmul(X, w) + b


def inference(X):
    # X is a placeholder variable
    w = tf.Variable(tf.random_normal([3, 3], stddev=0.01), name='weights')
    b = tf.Variable(tf.random_normal([3], stddev=0.01), name='biases')
    return model(X, w, b), w, b


def loss(y_hat, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_hat, y))


def train(loss):
    return tf.train.GradientDescentOptimizer(0.05).minimize(loss)


def predict(X, w, b):
    return tf.argmax(model(X, w, b), axis=1)


def run_training(x, y):
    data = Data(x, y)
    num_epochs = 2000
    
    X = data.Xtr
    Y = data.Ytr
    y_hat, w, b = inference(X)
    loss_op = loss(y_hat, Y)
    train_op = train(loss_op)
    
    Xte = data.Xte
    predict_op = predict(Xte, w, b)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for step in xrange(num_epochs):
            _, loss_value = sess.run([train_op, loss_op], feed_dict={X: data.xtr, Y: data.ytr})

            if step % 100 == 0:
                w_value, b_value = sess.run([w, b])
                print('Step %d: loss = %.2f' % (step, loss_value))
                predictions = sess.run(predict_op, feed_dict={Xte: data.xte})
                print w_value, b_value
                print predictions


if __name__ == '__main__':

    N = 2000
    w1 = np.array([[0.5, -1., -2.]])
    b1 = 0.1

    w2 = np.array([[-0.5, 3, 2.]])
    b2 = -0.5

    w3 = np.array([[0.1, -.3, 0.4]])
    b3 = 0

    x = np.random.randn(N, w1.shape[1])
    logits = np.hstack((np.dot(x, w1.T) + b1, np.dot(x, w2.T) + b2, np.dot(x, w3.T) + b3))
    y = np.asarray(map(lambda x: np.random.multinomial(1, x), softmax(logits)))
    run_training(x, y)
