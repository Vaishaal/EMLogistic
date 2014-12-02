from sklearn import cross_validation, linear_model
from scipy.special import expit as sigmoid
import numpy as np

class EMLogistic(object):
    def __init__(self):
        self.decision = 0.5
        self.clf = None
    def predict(self,X):
        return self.clf.predict(X)
    def fit(self,X,Y,max_iter=100):
        '''
            @param X is design matrix 
            @param Y is matrix where each column represents an estimators estimates
            for a particular row in X
        '''
        mu = (Y.sum(axis=0)/float(Y.shape[0]))[:,np.newaxis]
        converged = False
        i = 0
        while(not converged):
            alpha = ((mu*Y.T).sum(axis=0)/mu.sum())
            beta = ((1-mu)*(1-Y.T)).sum(axis=0)/(1-mu).sum()
            logistic_labels = mu > self.decision
            tmp_clf = linear_model.LogisticRegression()
            tmp_clf.fit(X, logistic_labels.ravel())
            w = tmp_clf.coef_
            p = sigmoid(X.dot(w.T))
            a = (np.power(alpha[:,np.newaxis],Y).prod(axis=0) * np.power((1 - alpha)[:,np.newaxis], 1 - Y).prod(axis=0))[:,np.newaxis]
            b = (np.power(beta[:,np.newaxis],1 - Y).prod(axis=0) * np.power((1 - beta)[:,np.newaxis], Y).prod(axis=0))[:,np.newaxis]
            mu = (a*p)/((a*p) + b*(1 - p))
            i+=1
            if (i >= max_iter):
                converged = True
                self.clf = tmp_clf
                self.labels = logistic_labels.ravel()
