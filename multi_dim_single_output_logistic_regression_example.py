# lets try linear regression
import tensorflow as tf
import numpy as np

num_epochs = 2000

def sigmoid(x):
    return 1./(1 + np.exp(-x))

def model(X, w, b):
    return tf.matmul(X, w) + b


def inference(X):
    # X is a placeholder variable
    w = tf.Variable(tf.random_normal([2, 1], stddev=0.01), name='weights')
    b = tf.Variable(tf.random_normal([1], stddev=0.01), name='biases')
    return model(X, w, b), w, b


def loss(y_hat, y):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_hat, y))


def train(loss):
    return tf.train.GradientDescentOptimizer(0.05).minimize(loss)


def predict(X, w, b):
    return tf.sigmoid(model(X, w, b))


def run_training(x, y):
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
                print (y * -np.log(p) + (1 - y) * -np.log(1 - p)).mean()
                print w_value, b_value


if __name__ == '__main__':

    N = 1000
    w = np.array([[0.5, -1.]])
    b = 0.1

    x = np.random.randn(N, w.shape[1])
    y = np.random.binomial(1, sigmoid(np.dot(x, w.T) + b))
    run_training(x, y)
