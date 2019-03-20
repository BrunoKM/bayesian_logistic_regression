import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn


def split_data(X, y, train_frac=0.8, seed=1):
    # Reseed the random state
    np.random.seed(seed)

    # Perform a train/test split of the data
    num_examples = y.shape[0]
    num_train_examples = int(train_frac * num_examples)
    assert num_train_examples > 0

    shuffled_idxs = np.random.permutation(num_examples)
    train_idxs = shuffled_idxs[:num_train_examples]
    test_idxs = shuffled_idxs[num_train_examples:]
    X_train, y_train = X[train_idxs], y[train_idxs]
    X_test, y_test = X[test_idxs], y[test_idxs]
    return X_train, y_train, X_test, y_test, train_idxs


def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def log_likelihood(y_true, y_pred):
    return y_true * np.log(y_pred) + (1 - y_true) * np.log(1.0 - y_pred)


def avg_log_likelihood(y_true, y_pred):
    return log_likelihood(y_true, y_pred).mean()


def expand_inputs(l, X, Z):
    X2 = np.sum(X**2, 1)
    Z2 = np.sum(Z**2, 1)
    ones_Z = np.ones(Z.shape[0])
    ones_X = np.ones(X.shape[0])
    r2 = np.outer(X2, ones_Z) - 2 * np.dot(X, Z.T) + np.outer(ones_X, Z2)
    return np.exp(-(0.5 / l**2) * r2)


def transform_to_rbf(data, radial_basis, width=0.01, add_bias_term=True):
    rbf_data = expand_inputs(width, data, radial_basis)
    if add_bias_term:
        rbf_data = np.hstack((np.ones([data.shape[0], 1]), rbf_data))
    return rbf_data


def log_determinant(mat):
    cholesky_l = np.linalg.cholesky(mat)
    # print(cholesky_l)
    return 2*np.sum(np.log(np.diag(cholesky_l)))


def hard_prediction(probs):
    return np.where(probs > .5, 1, 0)


def confusion_matrix_norm(y_true, y_pred):
    conf_mat = sklearn.metrics.confusion_matrix(y_true, y_pred).astype(np.float64).T
    for i in range(conf_mat.shape[0]):
        conf_mat[i, :] = conf_mat[i, :] / np.sum(y_pred == i)
    return conf_mat
