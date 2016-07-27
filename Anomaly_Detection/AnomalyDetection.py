from __future__ import division


"""
Sample python code implementing an anomaly detection algorithm
"""


import numpy as np
import matplotlib.pyplot as plt

# Set a random seed for reproducibility
np.random.seed(12345)


def GaussianEstimates(X):
    """
    Compute gaussian estimates for each feature in the matrix X.
    The gaussian distribution is given by:
    f(x, mu, sigma^2) = (1/(sqrt(2*pi)*sigma))*exp(-(x - mu)^2/2*sigma^2)

    :param X: A matrix containing features for which Gaussian parameters will be estimated
    :return: estimated mu and sigma for each feature (column) of X
    """

    # estimates of the mu_i
    # simply apply the mean function to each column
    mu_est = np.apply_along_axis(np.mean, 0, X)

    # estimates for sigmas
    # ddof is an argument to np.var forces the use of N-1 instead of N in the calculation
    # of sigma: sigma_i = sum ((x^(j) - mu_i)^2)/(n - 1)
    sigma_est = np.apply_along_axis(np.var, 0, X,ddof=0)

    return mu_est, sigma_est


def MultivariateGaussian(X, mu, sigma):
    """
    :param X: A numpy ndarray containing the data set for which we want
     to compute the probabilities
    :param mu: a vector containing estimates for mu
    :param sigma: a vector or matrix containing estimates for sigma
    :return: a vector containing likelihood estimates for each data point in X
    """
    def func(x, y):
        return np.dot(x, y)

    try:
        if np.shape(sigma)[1] == 1 or np.shape(sigma)[0] == 1:
            print "Sigma is either a column or a row vector"
    except IndexError:
        # Since we need a matrix for the vectorised  implementation, we will convert it to a
        # diagonal matrix under the assumption that the our features are independent
        sigma = np.diag(sigma)

    X = X - mu
    k = len(mu)

    sigma_inv = np.linalg.pinv(sigma)

    # X_(transpose) * sigma_inv
    # Note that elements in X are already transposed (row vectors), so no further transposition
    # needed
    a = np.apply_along_axis(func, 1, X, sigma_inv)

    p = np.power(2 * np.pi, -k / 2) * np.power(np.linalg.det(sigma), -0.5) * np.exp(np.sum(-0.5 * a * X, 1))

    return p


def VisualizeData(X, mu, sigma):
    """
    plot the data in X along with the fit.
    Note that X is a two column vector
    :param X:
    :param mu:
    :param sigma:
    :return:
    """
    np.seterr(over='ignore')
    x1, x2 = np.meshgrid(np.arange(0, 35, 0.5), np.arange(0, 35, 0.5))
    z1 = np.asarray(x1).reshape(-1)

    mat = np.zeros([len(z1), 2])
    mat[:, 0] = np.asarray(x1).reshape(-1)
    mat[:, 1] = np.asarray(x2).reshape(-1)

    Z = MultivariateGaussian(mat, mu, sigma)
    Z = np.reshape(Z, np.shape(x1))

    x = [10 ** x for x in np.arange(-20, 0, 3)]

    plt.figure(1)

    plt.scatter(X[:, 0], X[:, 1], c=None, s=25, alpha=None, marker="+")

    plt.contour(x1, x2, Z, x)
    plt.show()


def VisualizeFit(Xval, pval, epsilon, mu, sigma):
    """
    Visualize the fitter data
    :param Xval: the validation data set (only the first two columns are used)
    :param pval: A vector containing probabilities for example data in Xval
    :param mu: Estimate for the mean, using the training data
    :param sigma: Estimate for the variance, using the training data
    :return:
    """
    np.seterr(over='ignore')
    x1, x2 = np.meshgrid(np.arange(0, 35, 0.5), np.arange(0, 35, 0.5))
    z1 = np.asarray(x1).reshape(-1)

    mat = np.zeros([len(z1), 2])
    mat[:, 0] = np.asarray(x1).reshape(-1)
    mat[:, 1] = np.asarray(x2).reshape(-1)

    Z = MultivariateGaussian(mat, mu, sigma)
    Z = np.reshape(Z, np.shape(x1))

    x = [10 ** x for x in np.arange(-20, 0, 3)]

    plt.figure(1)

    plt.scatter(Xval[:, 0], Xval[:, 1], c=None, s=25, alpha=None, marker="+")

    points = np.where(pval < epsilon)
    plt.scatter(Xval[:, 0][points], Xval[:, 1][points], s=50, marker='+', color='red')
    plt.contour(x1, x2, Z, x)

    plt.show()


def SelectEpsilon(pval, yval):
    """
    Select the epsilon value for which the have the highest F_1 score on the validation
    set
    :param pval: vector of probabilities computed from the validation set
    :param yval: a vector containing the ground truth for the validation set
    :return: the best epsilon, the corresponding f_1 score
    """
    bestEpsilon = 0
    bestF1 = 0
    f1 = 0

    step = (max(pval) - min(pval)) / 1000;

    for epsilon in np.arange(min(pval), max(pval), step):
        predictions = map(lambda x: int(x < epsilon), pval)

        tp = sum(map(lambda x, y: int(x == 1 and y == 1), predictions, yval))

        fp = sum(map(lambda x, y: int(x == 1 and y == 0), predictions, yval))

        fn = sum(map(lambda x, y: int(x == 0 and y == 1), predictions, yval))

        if (tp + fp) != 0 and (tp + fn) != 0:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2*precision*recall/(precision + recall)

            if f1 > bestF1:
                bestF1 = f1
                bestEpsilon = epsilon
    return bestEpsilon, bestF1


def main():
    X = np.genfromtxt("train_features.csv", delimiter=',', skip_header=0)
    Xval = np.genfromtxt("validation_features.csv", delimiter=',', skip_header=0)
    yval = np.genfromtxt("validation_response.csv", delimiter=',', skip_header=0)

    # Estimates for mu and sigma from the train data
    mu, sigma = GaussianEstimates(X)
    # train the model on training data
    p = MultivariateGaussian(X, mu, sigma)

    # Get probabilities for each validation example
    pval = MultivariateGaussian(Xval, mu, sigma)
    # select the best epsilon along with the associated F1 score
    epsilon, f1score = SelectEpsilon(pval, yval)

    #VisualizeFit(Xval, pval, epsilon, mu, sigma)

    # Load a new data set and run the algorithm on it

    X = np.genfromtxt("train2_features.csv", delimiter=',', skip_header=0)
    Xval = np.genfromtxt("validation2_features.csv", delimiter=',', skip_header=0)
    yval = np.genfromtxt("validation2_response.csv", delimiter=',', skip_header=0)

    mu, sigma = GaussianEstimates(X)
    # Validation set
    pval = MultivariateGaussian(Xval, mu, sigma)
    epsilon, f1score = SelectEpsilon(pval, yval)

    # Get the number of outliers; that is potential anomalies

    print "Best epsilon found using cross validation: ", epsilon
    print "Best F1 score associated to the above epsilon: ", f1score
    print "Number of outliers found: ", len(np.where(pval < epsilon))



if __name__ == '__main__':
    main()




