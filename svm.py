from sklearn import svm
from sklearn.externals import joblib
import numpy as np

import os.path

class SVMModel:
    """
    Mô hình dự đoán, phân loại bot và real nodes, dựa trên SVM
    """

    def __init__(self):
        super().__init__()
        self.clf = None
        self.dumpFileName = 'svmmodel.joblib'

    def fit(self, nodes, labels):
        self.clf = svm.SVC()
        self.clf.fit(nodes, labels)
        joblib.dump(self.clf, self.dumpFileName)

    def predict(self, nodes):
        if self.clf == None:
            return None
        pred = self.clf.predict(nodes)
        return pred

    def load(self):
        if os.path.isfile(self.dumpFileName):
            self.clf = joblib.load(self.dumpFileName)