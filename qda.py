"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
2025-2026

Q3. Linear/Quadratic Discriminant Analysis (LDA/QDA).
"""
#Dominika PIECHOTA & Pawel DOROSZ

import numpy as np
import os

from sklearn.base import BaseEstimator, ClassifierMixin
from data import make_dataset1, make_dataset_breast_cancer
from plot import plot_boundary


class QuadraticDiscriminantAnalysis(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.lda = False
        self.classes = None
        self.means = {}
        self.priors = {}
        self.covs = {} # for QDA
        self.common_cov = None # for LDA

    def fit(self, X, y, lda=False):
        """Fit a linear discriminant analysis model using the training set
        (X, y).

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        # Input validation

        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2 dimensional")
        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("The number of samples differs between X and y")

        self.lda = lda

        self.classes = np.unique(y)
        n_samples, n_features = X.shape

        for cla in self.classes:
            X_cla = X[y == cla]
            self.priors[cla] = len(X_cla) / n_samples
            self.means[cla] = np.mean(X_cla, axis=0)
            if not lda:
                self.covs[cla] = np.cov(X_cla, rowvar=False)

        # cmpute the common covariance matrix for LDA
        # to do this: first calculate the covariance matrix for each class,
        # and then take their weighted average.
        if lda:
            self.common_cov = np.zeros((n_features, n_features))
            for cla in self.classes:
                X_cla = X[y == cla]
                self.common_cov += np.cov(X_cla, rowvar=False) * (1 / (n_samples - len(self.classes)))

        return self

    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes, or the predict values.
        """

        probs = self.predict_proba(X)
        return self.classes[np.argmax(probs, axis=1)]

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """

        num_of_obs = X.shape[0]
        p = np.zeros((num_of_obs, len(self.classes)))
        if self.lda:
            common_cov_inv = np.linalg.inv(self.common_cov)
            for cla in self.classes:
                for i in range(num_of_obs):
                    x=X[i]
                    delta_k = x.T @ common_cov_inv @ self.means[cla] - 0.5 * self.means[cla].T @ common_cov_inv @ self.means[cla] + np.log(self.priors[cla])
                    class_idx = np.where(self.classes == cla)[0][0]
                    p[i, class_idx] = delta_k

        else:
            for cla in self.classes:
                cov_inv = np.linalg.inv(self.covs[cla])
                cov_det = np.linalg.det(self.covs[cla])
                for i in range(num_of_obs):
                    x=X[i]
                    delta_k = -0.5 * np.log(cov_det)-0.5 * (x - self.means[cla]).T @ cov_inv @ (x - self.means[cla]) + np.log(self.priors[cla])
                    class_idx = np.where(self.classes == cla)[0][0]
                    p[i, class_idx] = delta_k


        p = np.exp(p - np.max(p, axis=1, keepdims=True))
        p /= np.sum(p, axis=1, keepdims=True)

        return p


# SAVE A DECISION BOUNDARY PLOT TO THE FOLDER FOR EACH MODEL
def save_boundary_plots(folder, dataset_id, models, test_X, test_y,method, accuracy):
    os.makedirs(folder, exist_ok=True)
    boundary_path = os.path.join(
        folder, f"dataset{dataset_id}_{method}.pdf"
    )
    plot_boundary(boundary_path[:-4], models, test_X, test_y, title = f"Accuracy = {accuracy:.3f}")


if __name__ == "__main__":

    accuracies_lda=[]
    accuracies_qda=[]

    for i in range (1,6): # 5 generations of the synthetic dataset
        print(f"\n===DATASET {i}===")

        data_X, data_y = make_dataset1(1200, random_state=i+41) # manually change the random state to ensure each dataset is different
        train_X, test_X = data_X[:900], data_X[900:] # 1200 * 0.75 = 900
        train_y, test_y = data_y[:900], data_y[900:]

        # QDA
        model_qda = QuadraticDiscriminantAnalysis()
        model_qda.fit(train_X, train_y, lda=False)
        preds_qda = model_qda.predict(test_X)
        acc_qda = np.mean(preds_qda == test_y)
        accuracies_qda.append(acc_qda)
        # save to PDF
        boundary_folder = f"dataset_{i}_boundaries_qda"
        save_boundary_plots(boundary_folder, i, model_qda, test_X, test_y,"qda",acc_qda)
        print(f"📂 Boundary plot saved to {boundary_folder}")

        # LDA
        model_lda = QuadraticDiscriminantAnalysis()
        model_lda.fit(train_X, train_y, lda=True)
        preds_lda = model_lda.predict(test_X)
        acc_lda = np.mean(preds_lda == test_y)
        accuracies_lda.append(acc_lda)
        #save to PDF
        save_boundary_plots(boundary_folder, i, model_lda, test_X, test_y,"lda", acc_lda)
        print(f"📂 Boundary plot saved to {boundary_folder}")

        print(f"make_dataset1_QDA:  Test acc={acc_qda:.3f}")
        print(f"make_dataset1_LDA:  Test acc={acc_lda:.3f}")

    print("\n\n===MEAN ACCURACIES AND STANDARD DEVIATIONS===")
    std_qda = np.std(accuracies_qda)
    mean_qda = np.mean(accuracies_qda)
    std_lda = np.std(accuracies_lda)
    mean_lda = np.mean(accuracies_lda)
    print(f"make_dataset1 QDA: Avg test acc={mean_qda:.3f}  Std test acc={std_qda:.3f}")
    print(f"make_dataset1 LDA: Avg test acc={mean_lda:.3f}  Std test acc={std_lda:.3f}")
    print()

    """    ------
    X : array of shape [569, 30]
        The feature matrix of the dataset
    y : array of shape [569]
        The labels of the dataset
    """

    bc_test_acc_qda = []
    bc_test_acc_lda = []

    for i in range(1,6): # 5 generations of the breast cancer dataset
        print(f"\n===BREAST CANCER DATASET {i}===")

        data_cancer_X, data_cancer_y = make_dataset_breast_cancer(random_state=i+41) # manually change the random state to ensure each dataset is different
        train_X, test_X = data_cancer_X[:427], data_cancer_X[427:] # 569 * 0.75 = 427
        train_y, test_y = np.asarray(data_cancer_y[:427]), np.asarray(data_cancer_y[427:])

        # QDA
        model_cancer_qda = QuadraticDiscriminantAnalysis()
        model_cancer_qda.fit(train_X, train_y, lda=False)
        preds_qda = model_cancer_qda.predict(test_X)
        acc_model_cancer_qda = np.mean(preds_qda == test_y)
        bc_test_acc_qda.append(acc_model_cancer_qda)
        print(f"model_cancer_QDA Test acc={acc_model_cancer_qda:.3f} ")

        # LDA
        model_cancer_lda = QuadraticDiscriminantAnalysis()
        model_cancer_lda.fit(train_X, train_y, lda=True)
        preds_lda = model_cancer_lda.predict(test_X)
        acc_model_cancer_lda = np.mean(preds_lda == test_y)
        bc_test_acc_lda.append(acc_model_cancer_lda)
        print(f"model_cancer_LDA Test acc={acc_model_cancer_lda:.3f} ")

    print("\n\n===MEAN ACCURACIES AND STANDARD DEVIATIONS===")
    mean_acc_qda_bc = np.mean(bc_test_acc_qda)
    mean_acc_lda_bc = np.mean(bc_test_acc_lda)
    std_acc_qda_bc = np.std(bc_test_acc_qda)
    std_acc_lda_bc = np.std(bc_test_acc_lda)
    print(f"QDA for breast cancer:     Avg test acc = {mean_acc_qda_bc:.3f}  Std test acc = {std_acc_qda_bc:.3f}")
    print(f"LDA for breast cancer:     Avg test acc = {mean_acc_lda_bc:.3f}  Std test acc = {std_acc_lda_bc:.3f}")