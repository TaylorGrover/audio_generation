import numpy as np

def softmax(z):
    assert type(z) is np.ndarray
    exp = np.exp(z)
    if len(exp.shape) == 2:

        return exp / np.sum(exp, axis=1).reshape(-1, 1)
    return np.exp(z) / np.sum(exp)


