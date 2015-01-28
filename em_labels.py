import numpy as np
class EMLabels(object):
    def __init__(self, decision = 0.5):
        self.y = None
        self.decision = decision
    def fit(self, Y, max_iter=100):
        '''
            @param Y is matrix where each column represents an estimators estimates
            for a particular sample
            @param max_iter is the maximum amount of iterations for the EM algorithm
        '''
        mu = (Y.sum(axis=0)/float(Y.shape[0]))[:,np.newaxis]
        for i in range(max_iter):
            T1 = mu.ravel() > self.decision
            T0 = 1 - T1
            pos1 = (T1*Y).sum(axis=1)
            tpr = pos1/float(T1.sum())
            fnr =  1 - tpr
            neg0 = (T0*(Y==0)).sum(axis=1)
            tnr = neg0/float(T0.sum())
            fpr = 1 - tnr
            Pi = np.array([[tpr,fpr],[fnr, tnr]])
            p1 = T1.sum()/float(len(T1))
            p0 = 1 - p1
            a = (np.power(tpr[np.newaxis].T,Y)).prod(axis=0)*p1
            b = (np.power(fnr[np.newaxis].T,Y == 0)).prod(axis=0)*p1
            c = (np.power(tnr[np.newaxis].T,Y == 0)).prod(axis=0)*p0
            d = (np.power(fpr[np.newaxis].T,Y)).prod(axis=0)*p0
            pos = (a*b)/((a*b)+(c*d))
            neg = (c*d)/((a*b) + (c*d))
            self.y = pos > neg
            if np.all(mu == self.y):
                print "CONVERGED in {0} iterations".format(i)
                break
            mu = self.y
        return self