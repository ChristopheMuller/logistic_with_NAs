import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.stats import norm, multivariate_normal

import warnings

warnings.filterwarnings("ignore")



def angular_error_beta(beta0, beta1):
    return np.arccos(np.dot(beta0, beta1) / (np.linalg.norm(beta0) * np.linalg.norm(beta1)))

def mse_error_beta(beta0, beta1):
    return np.mean((beta0 - beta1) ** 2)


class Classification:
    def __init__(self, name=""):
        self.name = name  # Name will use for plot

    def classification_error(self, X, M, y):
        return np.mean(np.abs(self.predict(X, M) - y))  # Classes are either 0 or 1
    
    def probability_error(self, X, M, y_probs, l=2):
        pred_probs = self.predict_probs(X, M)
        return np.linalg.norm(y_probs - pred_probs, l) / len(y_probs)


class RegLog05imputation(Classification):
    def __init__(self, name="05.IMP"):
        super().__init__(name)

        self.can_predict = True
        self.return_beta = True

    def fit(self, X, M, y):
        Xp = X.copy()
        Xp[M == 1] = 0.5
        self.reg = LogisticRegression(penalty=None).fit(
            Xp, y
        )

    def predict_probs(self, X, M):
        Xp = X.copy()
        Xp[M == 1] = 0.5
        return self.reg.predict_proba(Xp)[:, 1]

    def predict(self, X, M):
        Xp = X.copy()
        Xp[M == 1] = 0.5
        return self.reg.predict(Xp)
    
    def angular_error(self, true_beta):
        beta_hat = self.reg.coef_.ravel()
        return angular_error_beta(true_beta, beta_hat)
    
    def mse_error(self, true_beta):
        beta_hat = self.reg.coef_.ravel()
        return mse_error_beta(true_beta, beta_hat)
    
    def return_params(self):
        return [self.reg.coef_.ravel().tolist(), self.reg.intercept_.tolist()]


class RegLog05Mimputation(Classification):

    def __init__(self, name="05.IMP.M"):
        super().__init__(name)

        self.can_predict = True
        self.return_beta = True

    def fit(self, X, M, y):
        Xp = X.copy()
        Xp[M == 1] = 0.5
        # concat X and M
        Xp = np.hstack((Xp, M))
        self.reg = LogisticRegression(penalty=None).fit(
            Xp, y
        )

    def predict_probs(self, X, M):
        Xp = X.copy()
        Xp[M == 1] = 0.5
        Xp = np.hstack((Xp, M))
        return self.reg.predict_proba(Xp)[:, 1]

    def predict(self, X, M):
        Xp = X.copy()
        Xp[M == 1] = 0.5
        Xp = np.hstack((Xp, M))
        return self.reg.predict(Xp)
    
    def angular_error(self, true_beta):
        beta_hat = self.reg.coef_.ravel()
        return angular_error_beta(true_beta, beta_hat)
    
    def mse_error(self, true_beta):
        beta_hat = self.reg.coef_.ravel()
        return mse_error_beta(true_beta, beta_hat)
    
    def return_params(self):
        return [self.reg.coef_.ravel().tolist(), self.reg.intercept_.tolist()]


class RegLogMeanimputation(Classification):


    def __init__(self, name="Mean.IMP"):
        super().__init__(name)

        self.can_predict = True
        self.return_beta = True


    def fit(self, X, M, y):
        Xp = X.copy()

        for j in range(X.shape[1]):
            mean_j = np.nanmean(X[:,j])
            Xp[M[:,j] == 1, j] = mean_j
        
        self.reg = LogisticRegression(penalty=None).fit(
            Xp, y
        )

    def predict_probs(self, X, M):
        Xp = X.copy()
        for j in range(X.shape[1]):
            Xp[M[:,j] == 1, j] = np.nanmean(X[:,j])
        return self.reg.predict_proba(Xp)[:, 1]

    def predict(self, X, M):
        Xp = X.copy()
        for j in range(X.shape[1]):
            Xp[M[:,j] == 1, j] = np.nanmean(X[:,j])
        return self.reg.predict(Xp)
    
    def angular_error(self, true_beta):
        beta_hat = self.reg.coef_.ravel()
        return angular_error_beta(true_beta, beta_hat)
    
    def mse_error(self, true_beta):
        beta_hat = self.reg.coef_.ravel()
        return mse_error_beta(true_beta, beta_hat)  

    def return_params(self):
        return [self.reg.coef_.ravel().tolist(), self.reg.intercept_.tolist()]

    

class RegLogMeanMimputation(Classification):


    def __init__(self, name="Mean.IMP.M"):
        super().__init__(name)

        self.can_predict = True
        self.return_beta = True


    def fit(self, X, M, y):
        Xp = X.copy()
        self.mean_cols = np.nanmean(X, axis=0)
        for j in range(X.shape[1]):
            mean_j = self.mean_cols[j]
            Xp[M[:,j] == 1, j] = mean_j

        # concat X and M
        Xp = np.hstack((Xp, M))
        
        self.reg = LogisticRegression(penalty=None).fit(
            Xp, y
        )

    def predict_probs(self, X, M):
        Xp = X.copy()
        for j in range(X.shape[1]):
            Xp[M[:,j] == 1, j] = self.mean_cols[j]

        Xp = np.hstack((Xp, M))

        return self.reg.predict_proba(Xp)[:, 1]

    def predict(self, X, M):
        Xp = X.copy()
        for j in range(X.shape[1]):
            Xp[M[:,j] == 1, j] = np.nanmean(X[:,j])

        Xp = np.hstack((Xp, M))

        return self.reg.predict(Xp)
    
    def angular_error(self, true_beta):
        beta_hat = self.reg.coef_.ravel()
        return angular_error_beta(true_beta, beta_hat)
    
    def mse_error(self, true_beta):
        beta_hat = self.reg.coef_.ravel()
        return mse_error_beta(true_beta, beta_hat)
    
    def return_params(self):
        return [self.reg.coef_.ravel().tolist(), self.reg.intercept_.tolist()]


    
class RegLogICEimputation(Classification):


    def __init__(self, name="ICE.IMP"):
        super().__init__(name)

        self.can_predict = True
        self.return_beta = True

    def fit(self, X, M, y):
        Xp = X.copy()
        self.imp_mean = IterativeImputer(random_state=0, sample_posterior=True)
        self.imp_mean.fit(Xp)
        Xp = self.imp_mean.transform(Xp)
        self.reg = LogisticRegression(penalty=None).fit(
           Xp, y
        )

    def predict(self, X, M):
        Xp = X.copy()
        Xp = self.imp_mean.transform(Xp)
        return self.reg.predict(Xp, M)
    
    def predict_probs(self, X, M):
        Xp = X.copy()
        Xp = self.imp_mean.transform(Xp)
        return self.reg.predict_proba(Xp)[:, 1]
    
    def angular_error(self, true_beta):
        beta_hat = self.reg.coef_.ravel()
        return angular_error_beta(true_beta, beta_hat)
    
    def mse_error(self, true_beta):
        beta_hat = self.reg.coef_.ravel()
        return mse_error_beta(true_beta, beta_hat)
    
    def return_params(self):
        return [self.reg.coef_.ravel().tolist(), self.reg.intercept_.tolist()]


class RegLogICEMimputation(Classification):


    def __init__(self, name="ICE.IMP.M"):
        super().__init__(name)

        self.can_predict = True
        self.return_beta = True


    def fit(self, X, M, y):
        Xp = X.copy()
        self.imp_mean = IterativeImputer(random_state=0, sample_posterior=True)
        self.imp_mean.fit(Xp)
        Xp = self.imp_mean.transform(Xp)
        # concat X and M
        Xp = np.hstack((Xp, M))
        self.reg = LogisticRegression(penalty=None).fit(
           Xp, y
        )

    def predict(self, X, M):
        Xp = X.copy()
        Xp = self.imp_mean.transform(Xp)
        Xp = np.hstack((Xp, M))
        return self.reg.predict(Xp, M)
    
    def predict_probs(self, X, M):
        Xp = X.copy()
        Xp = self.imp_mean.transform(Xp)
        Xp = np.hstack((Xp, M))
        return self.reg.predict_proba(Xp)[:, 1]
    
    def angular_error(self, true_beta):
        beta_hat = self.reg.coef_.ravel()
        return angular_error_beta(true_beta, beta_hat)
    
    def mse_error(self, true_beta):
        beta_hat = self.reg.coef_.ravel()
        return mse_error_beta(true_beta, beta_hat)
    
    def return_params(self):
        return [self.reg.coef_.ravel().tolist(), self.reg.intercept_.tolist()]

class RegLogICEYimputation(Classification):
    def __init__(self, name="ICEY.IMP"):
        super().__init__(name)

        self.can_predict = True
        self.return_beta = True
    def fit(self, X, M, y):
        Xp = X.copy()
        data = np.hstack((X, y.reshape(-1, 1)))
        self.imp_mean = IterativeImputer(random_state=0, sample_posterior=True)
        self.imp_mean.fit(data)
        data_imputed = self.imp_mean.transform(data)
        Xp = data_imputed[:, :-1]
        self.reg = LogisticRegression(penalty=None).fit(
            Xp, y
        ) 

    def predict(self, X, M):
        Xp = X.copy()
        data = np.hstack((X, np.zeros(X.shape[0]).reshape(-1, 1)))
        data[:,-1] = np.nan
        data_imputed = self.imp_mean.transform(data)
        Xp = data_imputed[:, :-1]
        return self.reg.predict(Xp)
    
    def predict_probs(self, X, M):
        Xp = X.copy()
        data = np.hstack((X, np.zeros(X.shape[0]).reshape(-1, 1)))
        data[:,-1] = np.nan
        data_imputed = self.imp_mean.transform(data)
        Xp = data_imputed[:, :-1]
        return self.reg.predict_proba(Xp)[:, 1]

    def angular_error(self, true_beta):
        beta_hat = self.reg.coef_.ravel()
        return angular_error_beta(true_beta, beta_hat)
    
    def mse_error(self, true_beta):
        beta_hat = self.reg.coef_.ravel()
        return mse_error_beta(true_beta, beta_hat)

    def return_params(self):
        return [self.reg.coef_.ravel().tolist(), self.reg.intercept_.tolist()]


class RegLogICEYMimputation(Classification):
    def __init__(self, name="ICEY.IMP.M"):
        super().__init__(name)

        self.can_predict = True
        self.return_beta = True
    def fit(self, X, M, y):
        Xp = X.copy()
        data = np.hstack((X, y.reshape(-1, 1)))
        self.imp_mean = IterativeImputer(random_state=0, sample_posterior=True)
        self.imp_mean.fit(data)
        data_imputed = self.imp_mean.transform(data)
        Xp = data_imputed[:, :-1]
        Xp = np.hstack((Xp, M))
        self.reg = LogisticRegression(penalty=None).fit(
            Xp, y
        ) 

    def predict(self, X, M):
        Xp = X.copy()
        data = np.hstack((X, np.zeros(X.shape[0]).reshape(-1, 1)))
        data[:,-1] = np.nan
        data_imputed = self.imp_mean.transform(data)
        Xp = data_imputed[:, :-1]
        Xp = np.hstack((Xp, M))
        return self.reg.predict(Xp)
    
    def predict_probs(self, X, M):
        Xp = X.copy()
        data = np.hstack((X, np.zeros(X.shape[0]).reshape(-1, 1)))
        data[:,-1] = np.nan
        data_imputed = self.imp_mean.transform(data)
        Xp = data_imputed[:, :-1]
        Xp = np.hstack((Xp, M))
        return self.reg.predict_proba(Xp)[:, 1]

    def angular_error(self, true_beta):
        beta_hat = self.reg.coef_.ravel()
        return angular_error_beta(true_beta, beta_hat)
    
    def mse_error(self, true_beta):
        beta_hat = self.reg.coef_.ravel()
        return mse_error_beta(true_beta, beta_hat)

    def return_params(self):
        return [self.reg.coef_.ravel().tolist(), self.reg.intercept_.tolist()]

from sklearn.linear_model import LogisticRegression
import numpy as np

class CompleteCase(Classification):
    def __init__(self, name="CC"):
        super().__init__(name)
        self.can_predict = False
        self.return_beta = True
        self.used_not_all_columns = False
        self.remove_col = []
        self.reg = None  # Initialize logistic regression model

    def fit(self, X, M, y):
        """Fit complete-case logistic regression, handling edge cases."""
        
        self.can_predict = False
        self.return_beta = True
        self.used_not_all_columns = False
        self.remove_col = []
        self.reg = None  # Initialize logistic regression model

        n_samples, n_features = X.shape
        self.d = n_features
        self.X, self.M, self.y = X.copy(), M.copy(), y.copy()

        # Identify complete cases (rows without missing values)
        self.indices = np.all(self.M == 0, axis=1)

        if np.sum(self.indices) <= 1:  # If too few complete cases
            self.used_not_all_columns = True
            self.collapse = False

            while np.sum(self.indices) <= 1:  # Iteratively remove columns with most missing values
                print("Removing column with most missing values")
                na_count = np.sum(self.M, axis=0)

                if np.all(na_count == n_samples):  # If all columns have missing values
                    self.collapse = True
                    break

                col_with_most_na = np.argmax(na_count)
                self.remove_col.append(col_with_most_na)
                self.M = np.delete(self.M, col_with_most_na, axis=1)
                self.X = np.delete(self.X, col_with_most_na, axis=1)
                self.indices = np.all(self.M == 0, axis=1)

            if self.collapse:
                print("All columns had missing values. Imputing randomly.")
                self.X = np.random.normal(size=(n_samples, n_features))  # Random imputation

        # Ensure at least one sample of each class in the complete cases
        unique_classes, class_counts = np.unique(self.y[self.indices], return_counts=True)
        if len(unique_classes) == 1:  # If only one class remains
            for i in range(len(self.indices)):
                if self.indices[i]:
                    self.y[i] = 1 - self.y[i]  # Flip one sample to ensure both classes exist
                    break

        # Train logistic regression
        self.reg = LogisticRegression().fit(self.X[self.indices], self.y[self.indices])

    def predict(self, X, M):
        """Predict class labels."""
        if self.reg is None:
            raise ValueError("Model has not been trained. Call `fit` first.")
        return self.reg.predict(X)

    def predict_probs(self, X, M):
        """Predict probabilities."""
        if self.reg is None:
            raise ValueError("Model has not been trained. Call `fit` first.")
        return self.reg.predict_proba(X)[:, 1]

    def return_params(self):
        """Return model parameters, handling column removal cases."""
        if self.reg is None:
            return None
        
        beta = self.reg.coef_.ravel()
        intercept = self.reg.intercept_.tolist()

        if self.used_not_all_columns and (not self.collapse):
            # Restore beta to original feature space
            print(beta)
            print(self.remove_col)
            kept_cols = np.setdiff1d(np.arange(self.d), self.remove_col)
            print(kept_cols)
            full_beta = np.zeros(self.d)
            full_beta[kept_cols] = beta
            beta = full_beta
            print(beta)

        return [beta.tolist(), intercept]


