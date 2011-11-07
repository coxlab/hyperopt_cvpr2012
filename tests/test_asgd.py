import skdata.toy
from pythor3.wildwest.asgd_demo.asgd_one_vs_all import ASGDMultiClassSVM

def test_basic():
    dataset = skdata.toy.Digits()
    X, y = dataset.classification_task()

    svm = ASGDMultiClassSVM()
    svm.fit(X, y, num_epochs=10, num_samples_per_batch=10)

