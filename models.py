import numpy as np
import scipy.optimize
from util import logistic, log_likelihood, avg_log_likelihood, log_determinant, transform_to_rbf


# class RBFWrapper(object):
#     def __init__(self, base_model, rbf_width, radial_basis=None):
#         self.model = base_model
#         self.rbf_width = rbf_width
#         self.radial_basis = radial_basis
#
#     def transform_to_rbf(self, x, radial_basis=None):
#         if radial_basis is None:
#             assert self.radial_basis is not None
#             radial_basis = self.radial_basis
#         return transform_to_rbf(x, radial_basis, self.rbf_width)
#
#     def predict(self, x):
#         return self.model.predict(self.transform_to_rbf(x))


class LogisticClassifier(object):
    def __init__(self, input_size):
        self.input_size = input_size
        self.weights = np.zeros([input_size], dtype=np.float64)
        self.num_steps = 0
        return

    def update_weights(self, x, y, lr):
        grad_log_lik = x.T.dot((y - logistic(x.dot(self.weights))))
        self.weights += lr * grad_log_lik
        return

    def compute_avg_ll(self, x, y):
        """
        Compute the avg. log likelihood of the parameters given input x and labels y.
        """
        output_prob = self.predict(x)
        return avg_log_likelihood(y_true=y, y_pred=output_prob)

    def predict(self, x):
        return logistic(np.dot(x, self.weights))

    def predict_with_expanded(self, x, expand_func):
        return self.predict(expand_func(x))

    def hard_predict(self, x):
        return np.where(self.predict(x) > .5, 1, 0)


class LaplaceLogisticClassifier(object):
    """
    Bayesian Logistic Classifier with Laplace Approximation
    """
    def __init__(self, input_size, prior_mean=0., prior_var=1.):
        self.input_size = input_size
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.weights_map = None
        self.inv_covar = None
        self.covar = None
        # self.evidence = None
        # self.log_evidence = None
        return

    def fit_map(self, x, y, x_init=None):
        """
        Compute the weight values for the Maximum-a-posteriori of the weight posterior
        """
        if x_init is None:
             x_init = np.zeros([self.input_size], dtype=np.float64)

        def neg_log_posterior_func(weights):
            return -self.log_posterior_trunc(weights, x, y)

        def neg_log_posterior_jacobian(weights):
            return -self.log_posterior_jacobian(weights, x, y)

        res = scipy.optimize.minimize(neg_log_posterior_func, x_init, method='L-BFGS-B',
                                                   jac=neg_log_posterior_jacobian)
        if res['success']:
            self.weights_map = res['x']
        else:
            raise Exception('Unsuccessful optimisation:\n' + str(res['message']))
        # res = scipy.optimize.fmin_l_bfgs_b(neg_log_posterior_func, x0=x_init, fprime=neg_log_posterior_jacobian)
        # self.weights_map = res[0]
        return

    def log_posterior_trunc(self, weights: np.ndarray, x: np.ndarray, y: np.ndarray):
        """
        Compute the log-posterior excluding the log of Gaussian normalising constant on the prior (i.e. only
        keep the terms dependent on the weights). Used for optimisation purposes
        """
        ll_term = np.sum(logistic_log_likelihood(weights, x, y))
        prior_term = - 0.5 * np.sum((weights - self.prior_mean)**2) / self.prior_var
        return ll_term + prior_term

    def log_unnorm_posterior(self, weights: np.ndarray, x: np.ndarray, y: np.ndarray):
        """
        Compute the log-unnormalised-posterior
        """
        ll_term = np.sum(logistic_log_likelihood(weights, x, y))
        prior_term = -.5 * np.sum((weights - self.prior_mean)**2) / self.prior_var - \
                     .5 * self.input_size * np.log((2 * np.pi * self.prior_var))
        return ll_term + prior_term

    def log_posterior_jacobian(self, weights: np.ndarray, x: np.ndarray, y: np.ndarray):
        """
        Compute the grad of the log-posterior
        """
        grad_prior_term = - (weights - self.prior_mean) / self.prior_var
        grad_ll_term = logistic_ll_jacobian(weights, x, y)
        return grad_prior_term + grad_ll_term

    def calc_laplace_covariance(self, x):
        """
        Use the current MAP estimate to fit the covariance to the laplace approximation to the posterior.
        Note that this does not depend on the true labels
        """
        inv_covar = -logistic_ll_hessian(self.weights_map, x) + np.eye(self.input_size) / self.prior_var
        self.inv_covar = inv_covar
        # Make sure the matrix is positive definite todo
        return inv_covar

    def fit_laplace_approx(self, x, y):
        if self.weights_map is None:
            self.fit_map(x, y)
        self.calc_laplace_covariance(x)
        self.covar = np.linalg.inv(self.inv_covar)
        self.calc_evidence(x, y)

    def calc_evidence(self, x, y):
        """
        The approximate normalising constant from Laplace approx.
        """
        if self.inv_covar is None:
            raise ReferenceError("self.inv_covar is None. The inverse of covariance has not been calculated yet, but is"
                                 " needed for calc_evidence.")
        log_prob_of_data = np.sum(logistic_log_likelihood(self.weights_map, x, y)) - \
                           .5 * np.sum((self.weights_map - self.prior_mean) ** 2) / self.prior_var - \
                           .5 * self.input_size * np.log(self.prior_var) - \
                           .5 * log_determinant(self.inv_covar)
        prob_of_data = np.exp(log_prob_of_data)
        self.log_evidence = log_prob_of_data
        self.evidence = prob_of_data
        return log_prob_of_data, prob_of_data

    def bayesian_predict(self, x):
        pred_var = np.sum(x * np.dot(x, self.covar), axis=1)
        pred_mean = np.dot(x, self.weights_map)
        kappa = (1 + np.pi * pred_var / 8)**(-0.5)  # as defined in Bishop's book chapter 4
        bayes_predictions = logistic(pred_mean * kappa)
        return bayes_predictions

    def predict(self, x):
        return self.bayesian_predict(x)

    def hard_bayes_predict(self, x):
        return np.where(self.bayesian_predict(x) > .5, 1, 0)


# class RBFLaplaceLogisticClassifier(LaplaceLogisticClassifier):
#     def __init__(self, rbf_width, radial_basis, prior_mean=0., prior_var=1.):
#         input_size = radial_basis.shape[0] + 1
#         self.rbf_width = rbf_width
#         self.radial_basis = radial_basis
#         super().__init__(input_size, prior_mean=prior_mean, prior_var=prior_var)
#
#     def transform_data(self, x):
#         return transform_to_rbf(x, self.radial_basis, self.rbf_width, add_bias_term=True)
#
#     def fit_map(self, x, y, x_init=None):
#         super().fit_map(self.transform_data(x), y, x_init)
#
#     def fit_laplace_approx(self, x, y):
#         super().fit_laplace_approx(self.transform_data(x), y)
#
#     def predict(self, x):
#         super().predict(self.transform_data(x))
#
#     def bayesian_predict(self, x):
#         super().bayesian_predict(self.transform_data())
#
#     def calc_evidence(self, x, y):
#         super().calc_evidence(self.transform_data(x), y)


def logistic_log_likelihood(weights, x, y):
    output_probs = logistic(np.dot(x, weights))
    return log_likelihood(y_true=y, y_pred=output_probs)


def logistic_ll_jacobian(weights, x, y):
    return x.T.dot((y - logistic(x.dot(weights))))


def logistic_ll_hessian(weights, x):
    """Note the Hessian does not depend on the true labels."""
    output_probs = logistic(np.dot(x, weights))
    hessian = np.zeros([x.shape[1]] * 2, dtype=np.float64)
    for i in range(x.shape[0]):
        hessian -= output_probs[i] * (1 - output_probs[i]) * np.outer(x[i, :], x[i, :])
    return hessian

    # return -np.sum(output_probs * (1 - output_probs) * np.matmul(x[:, :, None], x[:, None, :]), axis=0)

