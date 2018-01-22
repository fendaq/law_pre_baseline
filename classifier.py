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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
import numpy as np
from gensim import matutils


inpath = '/disk/mysql/law_data/final_data/'
modelpath = 'model/'
classfierPath = 'model/classfier/'
accusation_path = 'accusation_list2.txt'
xf_path = 'xf_json.txt'

result = {}
accusation_list = []
dictionary = ''
xf_list = []
tfidf = ''

def init():
    #初始化一些全局变量
    global dictionary
    global tfidf
    global accusation_list
    global xf_list

    dictionary = corpora.Dictionary.load(modelpath + 'dictionary.model')
    tfidf = TfidfModel.load(modelpath + 'tfidf.model')
    fin = open(accusation_path, 'r')
    accusation_list = json.loads(fin.read())
    fin.close()
    fin = open(xf_path, 'r')
    xf_list = json.loads(fin.read()).keys()
    fin.close()

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
        self.zeros = 0
        self.model = LinearSVC()

    def havelabel(self, law):
        for m in law:
            try:
                if m[0] == self.num[0] and m[2] == self.num[1]:
                    return True
            except Exception as err:
                pass
        return False

    def addData(self, text, label):
        global dictionary
        global tfidf
        if self.havelabel(label):
            self.totalSize += 1
            if self.oneSize <= 400:
                self.label.append(1)
                self.oneSize += 1
            else:
                return
        else:
            self.zeroSize += 1
            #if self.zeroSize % 1000 == 0 and self.zeroSize <= 400:
            if self.zeroSize <= 500:
                self.zeros += 1
                self.label.append(0)
            else:
                return
        self.corpus.append(tfidf[dictionary.doc2bow(text.split())])

    def dataToMatrix(self):
        data = []
        rows = []
        cols = []
        for i, line in enumerate(self.corpus):
            for elem in line:
                rows.append(i)
                cols.append(elem[0])
                data.append(elem[1])
        self.matrix = csr_matrix((data,(rows,cols)))
        #释放数据，减少内存使用？
        self.corpus.clear()


    def train(self):
        global result

        print(self.num)
        print('totalSize:', self.totalSize, 'readSize:', self.oneSize, 'zeroSize:', self.zeros)
        #self.addData()
        data = {'totalSize:', self.totalSize, 'readSize:', self.oneSize, 'zeroSize:', self.zeros}
        self.dataToMatrix()
        self.chisquare()

        x_train, x_test, y_train, y_test = train_test_split(self.matrix, self.label, test_size=0.2)
        self.model.fit(x_train, y_train)

        y_prediect = self.model.predict(x_test)
        qu = self.quality(y_prediect, y_test)
        data['quality'] = qu

        joblib.dump(self.model, str(self.num) + '.model')
        print(self.matrix.shape)


    def chisquare(self):
        self.matrix = SelectKBest(chi2, k = self.dim).fit_transform(self.matrix, self.label)

    def quality(self, y_prediect, y_test):
        microPre = metrics.precision_score(y_test, y_prediect, average='micro')
        macroPre = metrics.precision_score(y_test, y_prediect, average='macro')
        microRecall = metrics.recall_score(y_test, y_prediect, average='micro')
        macroRecall = metrics.recall_score(y_test, y_prediect, average='macro')
        f1 = metrics.f1_score(y_test, y_prediect, average='weighted')
        ans = {}
        ans['microPre'] = microPre
        ans['macroPre'] = macroPre
        ans['microRecall'] = microRecall
        ans['macroRecall'] = macroRecall
        ans['F1'] = f1

        print(classification_report(y_test, y_prediect, target_names=['0', '1']))

        return ans


arrayOfClassifier = []
numberOfClassifier = len(xf_list)
print('numberOfClassifier:', numberOfClassifier)
for num in xf_list:
    arrayOfClassifier.append(classifier(num))


def train(c):
    fileList = os.listdir(inpath)
    for file in fileList:
        fin = open(inpath + file, 'r')
        line = fin.readline()
        while line:
            if c.oneSize > 400 and c.zeros > 500:
                break
            line = json.loads(line)
            c.addData(line['content'], line['meta']['law'])
            line = fin.readline()
        fin.close()
        if c.oneSize > 400 and c.zeros > 500:
            break
    c.train()

for v in arrayOfClassifier:
    train(v)





'''
abc = classifier([2, 0])
fin = open(inpath + '0', 'r')
line = fin.readline()
print('begin read')
while line:
    line = json.loads(line)
    abc.addData(line['content'], line['meta']['law'])
    line = fin.readline()
print('end read')

abc.train()
'''

