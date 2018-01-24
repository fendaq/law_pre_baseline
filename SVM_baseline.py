# coding: utf8

import json
import os
from scipy.stats import chisquare
from sklearn.externals import joblib
from gensim import corpora
from gensim.corpora.textcorpus import TextCorpus
from gensim.corpora.mmcorpus import MmCorpus
from gensim.models.tfidfmodel import TfidfModel
from scipy.sparse import csr_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics

from abc import ABCMeta, abstractmethod

class predict():
    __metaclass__ = ABCMeta
    def __init__(self):
        self.loadModel()

    @abstractmethod
    def loadModel(self):
        pass


    @abstractmethod
    def predict(self, text):
        pass



accusationModelPath = ''
articleModelPath = ''


class predictAccusation(predict):
    def loadModel(self):
        #self.model = joblib.load('accusation.model')
        pass

