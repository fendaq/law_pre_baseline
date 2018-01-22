#author: xcj
#this program is used to train SVM classifier!

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
from gensim import matutils


inpath = '/disk/mysql/law_data/final_data/'
modelpath = 'model/'
accusation_path = 'accusation_list2.txt'

accusation_list = []
dictionary = ''
tfidf = ''

def init():
    #初始化一些全局变量
    global dictionary
    global tfidf
    global accusation_list

    dictionary = corpora.Dictionary.load(modelpath + 'dictionary.model')
    tfidf = TfidfModel.load(modelpath + 'tfidf.model')
    fin = open(accusation_path, 'r')
    accusation_list = json.loads(fin.read())

init()


class classifier():
    def __init__(self, num, dim = 2000):
        self.num = num   #罪名代号
        #self.accusationName = accusation_list[num] #罪名名称
        self.dim = dim  #降维之后的向量维数
        self.totalSize = 0
        self.corpus = []
        self.label = []
        self.oneSize = 0
        self.zeroSize = 0
        self.model = LinearSVC()

    def addData(self, text, label):
        global dictionary
        global tfidf
        if label == self.num:
            self.totalSize += 1
            if self.oneSize <= 300:
                self.label.append(1)
                self.oneSize += 1
            else:
                return
        else:
            self.zeroSize += 1
            if self.zeroSize % 1000 == 0 and self.zeroSize <= 400:
                self.label.append(0)
            else:
                return
        self.corpus.append(tfidf[dictionary.doc2bow(text.split())])

    def dataToMatrix(self):
        data = []
        rows = []
        cols = []
        for i, line in enumerate(self.data):
            for elem in line:
                rows.append(i)
                cols.append(elem[0])
                data.append(elem[1])
        self.matrix = csr_matrix((data,(rows,cols)))
        #释放数据，减少内存使用？


    def train(self):
        self.getData()
        self.dataToMatrix()
        self.chisquare()
        self.model.fit(self.matrix, self.label)
        joblib.dump(self.model, str(self.num) + '.model')


    def chisquare(self):
        self.matrix = SelectKBest(chi2, k = self.dim).fit_transform(self.matrix, self.label)




