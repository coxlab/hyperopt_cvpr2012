import skdata.toy
from asgd.naive_asgd import NaiveBinaryASGD

def test_naive_asgd():
    dataset = skdata.toy.Digits()
    X, y = dataset.classification_task()

    binary_y = y < 5

    svm = NaiveBinaryASGD()
    svm.fit(X, y)

