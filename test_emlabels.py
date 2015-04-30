import unittest
from load_mnist import *
from denoise import *
from em_labels import EMLabels

def test_emlabel_sanity():
    features_all_cv, labels_all_cv = load_small_train(int(2))
    labels_all_cv = (np.array(labels_all_cv) == 1)
    noisy_labels = []
    for i in range(int(30)):
        noisy_labels.append(np.copy(labels_all_cv))
    noisy_labels = np.array(noisy_labels)
    clf = EMLabels()
    clf.fit(noisy_labels)
    assert(np.all(clf.y == labels_all_cv))

def test_small_very_noisy():
    train_size = 1
    label_noise = 0.8
    labelers = 10
    _test_noise(train_size, label_noise, labelers)

def test_large_little_noisy():
    train_size = 5
    label_noise = 0.4
    labelers = 10
    _test_noise(train_size, label_noise, labelers)


def test_many_labelers_very_noisy():
    train_size = 5
    label_noise = 0.9
    labelers = 100
    _test_noise(train_size, label_noise, labelers)

def test_iteration_count():
    train_size = 5
    label_noise = 0.9
    labelers = 100
    max_iter = 1
    em = _test_noise(train_size, label_noise, labelers, max_iter=max_iter)
    assert(em.iterations == 1)

def _test_noise(trainsize,label_noise, labelers, *args, **kwargs):
    features_all_cv, labels_all_cv = load_small_train(int(trainsize))
    labels_all_cv = (np.array(labels_all_cv) == 1)
    errs = []
    noisy_labels = []
    for i in range(int(labelers)):
        noise = random.uniform(0,float(label_noise))
        noisy_label = noisify(noise,labels_all_cv)
        noisy_labels.append(noisy_label)
        err = missclass(noisy_label, labels_all_cv)
        errs.append(err)
        #logging.info("MSE is " + str(err))
    noisy_labels = np.array(noisy_labels)
    majority_vote_labels = majority_vote(noisy_labels)
    em = EMLabels(*args, **kwargs)
    em.fit(noisy_labels)
    em_labels = em.y

    em_err = missclass(em_labels, labels_all_cv)
    majority_err = missclass(majority_vote_labels, labels_all_cv)
    assert(em_err <= majority_err)
    return em




