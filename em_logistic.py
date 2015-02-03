from sklearn import cross_validation, linear_model
from scipy.special import expit as sigmoid
from sklearn.metrics import roc_auc_score
import numpy as np

class EMLogistic(object):
    def __init__(self, decision=0.5, X_val=None, Y_val=None):
        self.decision = decision
        self.clf = None
        self.X_val = X_val
        self.Y_val = Y_val
    def predict(self,X):
        return self.clf.predict(X)
    def label(self, Y):
        weights = (self.alpha + self.beta)/2
        modified_weights = (len(weights)*weights/np.sum(weights))
        return np.sum(Y * modified_weights, axis=1) > modified_weights.shape[0] * self.decision
    def fit(self,X,Y,max_iter=1000):
        '''
            @param X is design matrix 
            @param Y is matrix where each column represents an estimators estimates
            for a particular row in X
        '''
        mu = (Y.sum(axis=0)/float(Y.shape[0]))[:,np.newaxis]
        converged = False
        i = 0
        tmp_clf = linear_model.SGDClassifier(loss="log",class_weight={1:0.03}, warm_start=True)
        while(not converged):
            print ("Iteration: " + str(i))
            alpha = ((mu*Y.T).sum(axis=0)/mu.sum())
            beta = ((1-mu)*(1-Y.T)).sum(axis=0)/(1-mu).sum()
            logistic_labels = mu > self.decision
            print set(logistic_labels.ravel())
            tmp_clf.fit(X, logistic_labels.ravel())
            self.clf = tmp_clf
            self.labels = logistic_labels.ravel()
            self.alpha = alpha
            self.beta = beta
            if (self.X_val != None and self.Y_val != None):
                print "auc: {0}".format(roc_auc_score(self.label(self.Y_val), tmp_clf.decision_function(self.X_val)))
            w = tmp_clf.coef_
            p = sigmoid(X.dot(w.T))
            a = (np.power(alpha[:,np.newaxis],Y).prod(axis=0) * np.power((1 - alpha)[:,np.newaxis], 1 - Y).prod(axis=0))[:,np.newaxis]
            b = (np.power(beta[:,np.newaxis],1 - Y).prod(axis=0) * np.power((1 - beta)[:,np.newaxis], Y).prod(axis=0))[:,np.newaxis]
            new_mu = (a*p)/((a*p) + b*(1 - p))
            if (np.linalg.norm(new_mu - mu) < 10e-2):
                converged = True
                print "CONVERGED"
            else:
                mu = new_mu
            i+=1
            if (i >= max_iter):
                converged = True
