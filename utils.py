import tensorflow as tf
from sklearn.cross_validation import train_test_split
import numpy as np


def create_normal_variable(*args):
    return tf.Variable(tf.random_normal(args, stddev=0.01))

def set_seeds(seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)


class Data(object):

    def __init__(self, x, y, test_fraction=0.2):
        # hold back 20% for testing
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_fraction)

        self.xtr = x_train
        self.ytr = y_train
        self.xte = x_test
        self.yte = y_test

        self.X = tf.placeholder(tf.float32, shape=[None, x_train.shape[1]], name='X')
        self.Y = tf.placeholder(tf.float32, shape=[None, y_train.shape[1]], name='Y')