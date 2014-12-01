import argparse
import random

import multiprocessing
import numpy as np
import numpy.linalg as la
import scipy.io
import logging
import ipdb
from scipy.stats import bernoulli

from console_progress import ConsoleProgress

ACCEPTED_LOG_LEVELS = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'WARN']
DATA_PREFIX = 'mnist'

def normalize_features(features):
    features = np.array(features)
    averages = np.reshape(np.average(features, axis=0), (features.shape[1], 1))
    ones =  np.ones((features.shape[0], 1))
    average_matrix = ones.dot(averages.T)
    semi_norm = features - average_matrix
    norms  = np.reshape(np.linalg.norm(semi_norm, axis=0), (features.shape[1],1))
    norm_matrix = ones.dot(norms.T)
    norm_matrix[np.where(norm_matrix == 0)] = 1
    return_val = semi_norm/norm_matrix
    return return_val

def get_features(data, normalize=False):
    num_tests = data[0][0][0].shape[2]
    features = []
    labels = []
    for n in range(num_tests):
        image = data[0][0][0][:,:,n]
        label = data[0][0][1].reshape((1, num_tests))[0][n]
        features.append(image.flatten().tolist())
        labels.append(label)
    features = np.array(normalize_features(features))
    labels = np.array(labels)
    return (features, labels)

def load_mat_features(name):
    assert name in ('train', 'test')
    return get_features(scipy.io.loadmat(DATA_PREFIX + name + '.mat')[name])

def load_small_train(size):
    return get_features(scipy.io.loadmat(DATA_PREFIX+'train_small.mat')['train'][0][size])

def noisify(noise,data,prior=0.3):
    data = np.copy(data)
    indices = map(int,np.random.uniform(low=0.0, high=len(data)-1, size=len(data)*noise))
    noisy = scipy.stats.bernoulli.rvs(prior, len(data)*noise)
    data[indices] = noisy
    return data



def main():
    global DATA_PREFIX
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-prefix', dest='data_prefix', help='Base directory, should contain \"{size}\"\"{train, test}\".mat',
            default='mnist')
    parser.add_argument('--train-size', dest='trainsize', help='0-6',
            default=2)
    parser.add_argument('--log', dest='log', nargs='?', const='INFO',
            default='INFO', help='Set log level (default: WARN)')

    parser.add_argument('--labelers', dest='labelers', help="How many noisy labelers do you want?",
            default=10)
    parser.add_argument('--noise', dest='label_noise', help='How much label noise do you want',
            default=0.2)
    args = parser.parse_args()
    if args.log in ACCEPTED_LOG_LEVELS:
        logging.basicConfig(level=eval('logging.'+args.log))

    DATA_PREFIX = args.data_prefix
    if DATA_PREFIX[-1] != '/':
        DATA_PREFIX += '/'

    features_all_cv, labels_all_cv = load_small_train(int(args.trainsize))
    labels_all_cv = (np.array(labels_all_cv) == 1)
    errs = []
    for i in range(int(args.labelers)):
        diff = labels_all_cv - noisify(float(args.label_noise),labels_all_cv)
        err = np.sqrt(np.sum(diff * diff))
        errs.append(err)
        logging.info("MSE is " + str(err))
    logging.info("Average MSE is " + str(np.average(np.array(errs))))

if __name__=='__main__':
    main()
