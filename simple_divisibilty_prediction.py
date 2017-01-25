import numpy as np
from fractions import gcd
from multi_layer_nn_example import run_training

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

def binary_representation(n):
    MAX_DIGIT = 15
    assert n < 2**MAX_DIGIT
    return map(int, (bin(n).lstrip('0b').zfill(MAX_DIGIT)))

def reverse_binary_representation(l):
    return sum(map(lambda (e, c): c*2**e, zip(reversed(range(len(l))), l)))

def one_hot_encoding_decimal_digits(n):
    MAX_DIGIT = 15
    assert n < 2**MAX_DIGIT
    v = [0]* 10 * 4

    for i, digit in enumerate(map(int, str(n).zfill(4))):
        v[digit + 10*i] = 1
    return v

def reverse_one_hot_encoding(l):
    """
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
       0, 0, 0, 0]
    """
    v = ""
    for i in range(0, len(l), 10):
        one_hot_digit = l[i:i+10]
        v += str(one_hot_digit.index(1))
    return int(v)


def create_feature_vector(n):
    return binary_representation(n)


N = range(4000)
x = np.array([create_feature_vector(i) for i in N])

divisor = 15
y = np.array([n % divisor == 0 for n in N])[:, None]
w, b, test_data, test_predictions = run_training(x, y, n_hidden_nodes=100, n_layers=3, num_epochs=5000)

test_predictions = test_predictions[:,0]
ix = np.argsort(test_predictions)

test_predictions = test_predictions[ix]
test_data = test_data[ix]

for d, p in zip(test_data, test_predictions):
    print reverse_binary_representation(list(d)), list(d), reverse_binary_representation(list(d)) % divisor == 0,  p