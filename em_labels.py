import numpy as np
import logging

class EMLabels(object):
    def __init__(self, decision = 0.5, logger=None):
        self.y = None
        self.raw_y = None
        self.decision = decision
        if (logger == None):
            logger = logging.getLogger('auto-em')
        self.logger =  logger
        self.pi = None
    def fit(self, Y, max_iter=100):
        '''
            @param Y is a M x N matrix. N = number of samples, M = number of labellers.
            @param max_iter is the maximum amount of iterations for the EM algorithm
        '''
        # Initial guess for labels (majority vote)
        mu = (Y.sum(axis=0)/float(Y.shape[0]))[:,np.newaxis]

        # Threshold the floats into integer labels
        T1 = mu.ravel() > self.decision

        for i in range(max_iter):
            # BEGIN E STEP

            T0 = 1 - T1
            # Multiply T1 by Y to get a matrix which has a 1 exactly when our positive labels matched labellers positive labels
            pos1 = (T1*Y).sum(axis=1)
            tpr = pos1/float(T1.sum())
            fnr =  1 - tpr

            # Multiply T1 by (Y==0) to get a matrix which has a 1 exactly when our negative labels matched labellers negative labels
            neg0 = (T0*(Y==0)).sum(axis=1)
            tnr = neg0/float(T0.sum())
            fpr = 1 - tnr


            # Our prior on positive labels
            p1 = T1.sum()/float(len(T1))
            p0 = 1 - p1
            # END E STEP

            # BEGIN M STEP

            # Calculate p(labels = 1| data) using bayes theorem

            a = (np.power(tpr[np.newaxis].T,Y)).prod(axis=0)*p1
            b = (np.power(fnr[np.newaxis].T,Y == 0)).prod(axis=0)*p1

            c = (np.power(tnr[np.newaxis].T,Y == 0)).prod(axis=0)*p0
            d = (np.power(fpr[np.newaxis].T,Y)).prod(axis=0)*p0
            pos = (a*b)/((a*b)+(c*d))
            neg = (c*d)/((a*b) + (c*d))

            # pos = E[T1 = 1| Y]

            # TODO: This may need to be changed to use self.decision
            self.y = pos > neg
            del_y = np.sum(np.logical_xor(self.y,T1))
            self.logger.debug("Iteration {0}: del_y={1} ".format(i, del_y))


            # update y for next iteration
            self.tpr = tpr
            self.tnr = tnr
            self.p1 = p1
            self.raw_y = pos

            if np.all(T1 == self.y):
                print "CONVERGED in {0} iterations".format(i)
                print tpr
                print fpr
                break
            T1 = self.y
            # END M STEP
        self.iterations = i
        return self
    def predict(self, y):
        tpr = self.tpr
        fpr = 1 - self.tnr
        tnr = self.tnr
        fnr = 1 - self.tpr
        p0 = 1 - self.p1
        p1 = self.p1
        a = (np.power(tpr.T,y)).prod(axis=0)*p1
        b = (np.power(fnr.T,y == 0)).prod(axis=0)*p1
        c = (np.power(tnr.T,y == 0)).prod(axis=0)*p0
        d = (np.power(fpr.T,y)).prod(axis=0)*p0
        pos = (a*b)/((a*b)+(c*d))
        neg = (c*d)/((a*b) + (c*d))
        return pos > neg
