import argparse
import random

import multiprocessing
import numpy as np
import numpy.linalg as la
import scipy.io
import logging
import ipdb
from scipy.stats import bernoulli
from sklearn import cross_validation, linear_model

from console_progress import ConsoleProgress
from load_mnist import *



def majority_vote(labels):
    return np.sum(labels, axis=0) > labels.shape[0]*0.5

def mse(labels, true_labels):
    diff = labels - true_labels
    err = np.sqrt(np.sum(diff * diff))
    return err

def missclass(labels, true_labels):
    return np.sum(labels != true_labels)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-prefix', dest='data_prefix', help='Base directory, should contain \"{size}\"\"{train, test}\".mat',
            default='mnist')
    parser.add_argument('--train-size', dest='trainsize', help='0-6',
            default=6)
    parser.add_argument('--log', dest='log', nargs='?', const='INFO',
            default='INFO', help='Set log level (default: WARN)')

    parser.add_argument('--labelers', dest='labelers', help="How many noisy labelers do you want?",
            default=10)
    parser.add_argument('--noise', dest='label_noise', help='Maximum amount of label noise do you want',
            default=1)
    args = parser.parse_args()
    if args.log in ACCEPTED_LOG_LEVELS:
        logging.basicConfig(level=eval('logging.'+args.log))

    DATA_PREFIX = args.data_prefix
    if DATA_PREFIX[-1] != '/':
        DATA_PREFIX += '/'

    features_all_cv, labels_all_cv = load_small_train(int(args.trainsize))
    labels_all_cv = (np.array(labels_all_cv) == 1)
    errs = []
    noisy_labels = []
    for i in range(int(args.labelers)):
        noise = random.uniform(0,float(args.label_noise))
        noisy_label = noisify(noise,labels_all_cv)
        noisy_labels.append(noisy_label)
        err = missclass(noisy_label, labels_all_cv)
        errs.append(err)
        #logging.info("MSE is " + str(err))
    noisy_labels = np.array(noisy_labels)
    logging.info("Average Missclassification raw is " + str(np.average(np.array(errs))))
    majority_vote_labels = majority_vote(noisy_labels)
    logging.info("Missclassification majority vote is " + str(missclass(majority_vote_labels, labels_all_cv)))
    clf = linear_model.LogisticRegression()
    cv = cross_validation.ShuffleSplit(len(labels_all_cv), n_iter=3, test_size=0.3, random_state=0)
    majority_vote_cv_score = cross_validation.cross_val_score(clf,features_all_cv,majority_vote_labels,scoring="f1",cv=cv)
    logging.info("Majority vote cross-val score is " + str(majority_vote_cv_score))
    true_cv_score = cross_validation.cross_val_score(clf,features_all_cv,labels_all_cv,scoring="f1",cv=cv)
    logging.info("True cross-val score is " + str(true_cv_score))

if __name__=='__main__':
    main()
