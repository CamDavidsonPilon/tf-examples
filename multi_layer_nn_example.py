import tensorflow as tf
import numpy as np
from utils import set_seeds, Data, create_normal_variable


set_seeds(43)


def model(X, weights, biases, dropout_prob):
    n_layers = len(weights)
    output = tf.add(tf.matmul(X, weights['input']), biases['input'])
    output = tf.nn.dropout(tf.nn.relu(output), dropout_prob)

    for i in xrange(2, n_layers):
        output = tf.add(tf.matmul(output, weights['h%i' % i]), biases['b%i' % i])
        output = tf.nn.relu(output)

    output = tf.add(tf.matmul(output, weights['output']), biases['output'])
    return output, weights, biases


def inference(X, weights, biases):
    return model(X, weights, biases, 0.5)


def loss(y_hat, y):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_hat, y))


def train(loss):
    return tf.train.RMSPropOptimizer(learning_rate=0.0003, decay=0.8, momentum=0.4).minimize(loss)


def predict(X, weights, biases):
    return model(X, weights, biases, 1.0)[0]


def create_weights(n_input, n_layers, n_hidden_nodes, n_output):

    weights = {
        'h%i' % i: create_normal_variable(n_hidden_nodes, n_hidden_nodes)
        for i in xrange(2, n_layers)
    }
    weights['input'] = create_normal_variable(n_input, n_hidden_nodes)
    weights['output'] = create_normal_variable(n_hidden_nodes, n_output)

    biases = {
        'b%i' % i: create_normal_variable(n_hidden_nodes)
        for i in range(2, n_layers)
    }
    biases['input'] = create_normal_variable(n_output)
    biases['output'] = create_normal_variable(n_output)
    return weights, biases


def run_training(x, y, n_hidden_nodes=5, n_layers=2, num_epochs=5000, loss=loss):
    datasets = Data(x, y)
    n_input = x.shape[1]
    n_output = 1

    weights, biases = create_weights(n_input, n_layers, n_hidden_nodes, n_output)

    y_hat, fitted_weights, fitted_biases = inference(datasets.X, weights, biases)
    loss_op = loss(y_hat, datasets.Y)
    train_op = train(loss_op)
    predict_op = predict(datasets.X, weights, biases)
    predict_loss_op = loss(predict_op, datasets.Y)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for step in xrange(num_epochs):

            # Shuffle the data before each training iteration.
            p = np.random.permutation(range(len(datasets.xtr)))
            xtr, ytr = datasets.xtr[p], datasets.ytr[p]

            batch_size = 128
            for start in range(0, len(datasets.xtr), batch_size):
                end = start + batch_size
                sess.run(train_op, feed_dict={datasets.X: xtr[start:end], datasets.Y: ytr[start:end]})


            if step % 100 == 0:
                train_loss = sess.run(loss_op, feed_dict={datasets.X: xtr, datasets.Y: ytr})
                test_predictions, test_loss = sess.run([predict_op, predict_loss_op], feed_dict={datasets.X: datasets.xte, datasets.Y: datasets.yte})
                print('Step %d: training_loss = %.3f: test_loss = %.3f' % (step, train_loss, test_loss))

        print("Final")
        test_predictions, test_loss, fitted_weights, fitted_biases = sess.run([predict_op, predict_loss_op, fitted_weights, fitted_biases],
                                                                               feed_dict={datasets.X: datasets.xte, datasets.Y: datasets.yte})
        print('Step %d: training_loss = %.3f: test_loss = %.3f' % (step, train_loss, test_loss))

    return fitted_weights, fitted_biases, datasets.xte, test_predictions



if __name__ == '__main__':

    def sigmoid(x):
        return 1./(1 + np.exp(-x))


    N = 1000
    w = np.random.randn(1,25)
    b = 0.1

    x = np.random.randn(N, w.shape[1])
    p = sigmoid(x[:, 0])[:, None]*sigmoid(np.dot(x, w.T) + b)
    y = np.random.binomial(1, p)
    w, b = run_training(x, y, n_hidden=8)
