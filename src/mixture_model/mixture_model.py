import os
import h5py
import random
import numpy as np

from scipy.misc import logsumexp
from sklearn.mixture import GaussianMixture


HOME_DIR = os.path.expanduser("~")
DATA_DIR = os.path.join(HOME_DIR, "ml-final-project", "data", "msd_data_10.hdf5")


class MixtureModel(object):
    def __init__(self, num_classes, num_features, num_mixture_components):
        self.num_classes = num_classes
        self.num_features = num_features
        self.num_mixture_components = num_mixture_components

        self.max_iter = 50
        self.epsilon = 1e-3

        self.params = {
            "pi": [np.repeat(1 / k, k) for k in self.num_mixture_components],
            #"theta": [np.zeros((self.num_features, k)) for k in self.num_mixture_components]
            "mu": []
        }
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def pack_params(self, X, class_idx):
        pi, theta = self.fit(X[class_idx], class_idx)
        self.params["pi"][class_idx] = pi
        self.params["theta"][class_idx] = theta

    def classify(self, X):
        P = []
        pi = self.params["pi"]
        theta = self.params["theta"]
        for c in range(self.num_classes):
            _, Pc = self.find_P(X, pi[c], theta[c], self.epsilon)
            P.append(Pc)

        print(np.vstack(P))
        quit()
        return np.vstack(P).T.argmax(-1)

    def update_latent_posterior(self, X, pi, theta, num_mixture_components):
        t, Pc = self.find_P(X, pi, theta, self.epsilon)
        log_gamma = t - Pc[:, np.newaxis]
        gamma = np.exp(log_gamma)
        return gamma

    def update_pi(self, gamma):
        pi_c = np.sum(gamma, axis=0) / gamma.shape[0]
        return pi_c

    def update_theta(self, X, gamma):
        N_c = np.sum(gamma, axis=0)
        theta_c = (1 / N_c)[np.newaxis, :] * np.dot(X.T, gamma)
        return theta_c

    def find_P(self, X, pi, theta, epsilon):
        t = np.log(pi + epsilon)[np.newaxis, :] \
            + np.dot(X, np.log(theta + epsilon)) \
            + np.dot(1 - X, np.log(1 - theta + epsilon))

        return t, logsumexp(t, axis=1)

    def fit(self, X, class_idx):
        max_iter = self.max_iter
        eps = self.epsilon
        N = X.shape[0]
        pi = self.params["pi"][class_idx]
        theta = self.params["theta"][class_idx]
        num_mixture_components = self.num_mixture_components[class_idx]

        for i in range(num_mixture_components):
            unorm = np.sum(X[range(i, N, num_mixture_components), :], axis=0) \
                / np.size(X[range(i, N, num_mixture_components), :], 0)
            theta[:, i] = np.absolute(unorm) / (np.absolute(unorm).max() + eps)
        print(theta.max(), theta.min())

        for i in range(max_iter):
            gamma = self.update_latent_posterior(X, pi, theta, num_mixture_components)
            pi = self.update_pi(gamma)
            theta = self.update_theta(X, gamma)

        return pi, theta

    def train(self, X):
        for c in range(self.num_classes):
            self.pack_params(X, c)

    def val(self, X, acc=0, N=0):
        for c in range(self.num_classes):
            classifications = self.classify(X[c])
            for classification in classifications:
                self.confusion_matrix[c, classification] += 1
            acc += np.sum((classifications == c).astype(np.int32))
            N += X[c].shape[0]

        return (acc / N)

def load_data(classes=10):
    X_train = {i: [] for i in range(classes)}
    X_val = {i: [] for i in range(classes)}

    with h5py.File(DATA_DIR, "r") as f:
        stop_val = np.zeros(f["/data"][0].shape)
        num_features = stop_val.shape[0]
        i = 0
        while not (f["/data"][i] == stop_val).all():
            """
            if random.random() < 0.1:
                X_val[f["/labels"][i]].append(f["/data"][i])
            else:
                X_train[f["/labels"][i]].append(f["/data"][i])
            """
            i += 1

        j = int(0.9 * i)
        X_train = f["/data"][:j, :]
        y_train = f["/labels"][:j]
        X_val = f["/data"][j:i, :]
        y_val = f["/labels"][j:i]

    """
    for c in range(classes):
        X_train[c] = np.array(X_train[c])
        X_val[c] = np.array(X_val[c])
    """

    return X_train, y_train, X_val, y_val, num_features


def main():

    X_train, y_train, X_val, y_val, num_features = load_data()
    num_classes = 10

    gmm = GaussianMixture(n_components=num_classes)
    gmm.means_ = np.array([X_train[y_train == c].mean(axis=0) for c in range(num_classes)])
    gmm.fit(X_train)
    y_train_pred = classifier.predict(X_train)
    train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
    print(train_accuracy)

    """
    for i in range(1):
        num_mixture_components = np.random.randint(2, 15, num_classes)
        print("COMPONENTS: " + " ".join(str(i) for i in num_mixture_components))
        mm = MixtureModel(num_classes, num_features, num_mixture_components)
        mm.train(X_train)
        acc = mm.val(X_train)
        print("ACCURACY ON VALIDATION", acc)
        print(mm.confusion_matrix)
    """


if __name__ == "__main__":
    main()
