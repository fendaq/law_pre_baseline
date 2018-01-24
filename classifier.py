# coding: utf8
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
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
from scipy.sparse import vstack
from gensim import matutils
import heapq


#inpath = '/disk/mysql/law_data/final_data/'
inpath = '/home/guozhipeng/law_data/goodData/'
outpath = 'predict_data/'
modelpath = 'model/'
classfierPath = 'model/classfier/'
lawPath = 'law_result1.txt'
#accusation_path = 'accusation_list2.txt'
#xf_path = 'xf_json.txt'

result = {}
dictionary = ''
tfidf = ''
law_list = []
law_dic = {}
tobe_law = {}
#xf_list = []


def init():
    #初始化一些全局变量
    global dictionary
    global tfidf
    global accusation_list
    global law_list

    dictionary = corpora.Dictionary.load(modelpath + 'dictionary.model')
    tfidf = TfidfModel.load(modelpath + 'tfidf.model')

    fin = open(lawPath, 'r')
    line = fin.readline()
    while line:
        line = line.split()
        law_list.append([int(line[0]), int(line[1])])
        line = fin.readline()
    fin.close()
    for i, v in enumerate(law_list):
        law_dic[str(v)] = i
        tobe_law[i] = v

init()


class classifier():
    def __init__(self, dim = 2000):
        #self.num = num   #罪名代号
        #self.accusationName = accusation_list[num] #罪名名称

        self.dim = dim  #降维之后的向量维数
        self.corpus = []
        self.label = []
        self.testSize = 0
        self.trainSize = 0
        self.model = OneVsRestClassifier(LinearSVC().set_params(probability=True))

        '''
        self.totalSize = 0
        self.testCorpus = []
        self.testLabel = []
        self.oneSize = 0
        self.zeroSize = 0
        self.testOneSize = 0
        self.testZeroSize = 0
        #self.model = LinearSVC()
        '''


    def addData(self, line, test = False):
        global dictionary
        global tfidf
        global law_dic

        text = []
        for s in line['content']:
            text += s
        self.corpus.append(tfidf[dictionary.doc2bow(text)])
        #for law predict
        self.label.append(law_dic[str(line['meta']['law'])])

        #for accusation predict
        #self.label.append(line['meta']['crit'])

        #for time predict
        #self.label.append(line['meta']['time'])

        if test:
            self.testSize += 1
        else:
            self.trainSize += 1

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
        self.corpus.clear()  # 释放数据，减少内存使用


    def train(self):
        #global result

        #print(self.num)

        self.dataToMatrix()
        self.chisquare()
        self.train_test_split()

        print('begin to train')
        self.model.fit(self.trainx, self.trainy)
        print('train end')
        joblib.dump(self.model, 'model/law_predict.model')
        #joblib.dump(self.model, classfierPath + str(self.num) + '.model')

    def test(self):
        y_predict = self.model.predict(self.testx)

        print(classification_report(self.testy, y_predict))
        print('microPrecision:', metrics.precision_score(self.testy, y_predict, average='micro'))
        print('microRecall:', metrics.recall_score(self.testy, y_predict, average='micro'))
        print('microF1:', metrics.f1_score(self.testy, y_predict, average='micro'))
        print('macroPrecision:', metrics.precision_score(self.testy, y_predict, average='macro'))
        print('macroRecall:', metrics.recall_score(self.testy, y_predict, average='macro'))
        print('macroF1:', metrics.f1_score(self.testy, y_predict, average='macro'))

        #data = {'totalSize': self.totalSize, 'oneSize': self.oneSize, 'zeroSize:': self.zeroSize, 'testSize': self.testOneSize + self.testZeroSize, 'testOneSize': self.testOneSize}
        #print(data)
        #y_predict = self.model.predict(self.testMatrix)
        #qu = self.quality(y_predict, self.testLabel)
        #data['quality'] = qu
        #print(data)

    def getLine(self):
        for i in range(20):
            fin = open(inpath + str(i), 'r')
            line = fin.readline()
            while line:
                yield [i, json.loads(line)]
                line = fin.readline()
            fin.close()


    def predict(self, k = 2):
        global tobe_law

        #fileIndex = 0
        #fout = open(outpath + str(fileIndex), 'w')
        ally = self.model.decision_function(vstack([self.trainx, self.testx]))
        print(type(ally))
        print(ally.shape)
        num = 0
        for i in range(20):
            fin = open(inpath + str(i), 'r')
            fout = open(outpath + str(i), 'w')
            line = fin.readline()
            while line:
                line = json.loads(line)
                y = ally[num].tolist()
                kLargestList = heapq.nlargest(k, y)
                k1 = y.index(kLargestList[0])
                k2 = y.index(kLargestList[1])
                line['meta']['top2law'] = [tobe_law[k1], tobe_law[k2]]
                print(json.dumps(line), file = fout)
                #yield [i, json.loads(line)]
                line = fin.readline()
                num += 1
            fin.close()
            fout.close()

        '''
        for y in ally:
            y = y.tolist()
            kLargestList = heapq.nlargest(k, y)
            k1 = y.index(kLargestList[0])
            k2 = y.index(kLargestList[1])
            data = self.getLine()
            data[1]['meta']['top2law'] = [tobe_law[k1], tobe_law[k2]]
            if data[0] != fileIndex:
                fout.close()
                fileIndex += 1
                fout = open(outpath + str(fileIndex), 'w')
            print(json.loads(data[1]), file=fout)
        '''
        '''
        print(self.trainx.shape)
        print(self.testx.shape)
        allx = np.vstack((self.trainx, self.testx))
        print(allx.shape)
        fileIndex = 0
        fout = open(outpath + str(fileIndex), 'w')
        for x in allx:
            predict_result = self.model.predict_proba(x)
            kLargestList = heapq.nlargest(k, predict_result)
            k1 = predict_result.index(kLargestList[0])
            k2 = predict_result.index(kLargestList[1])
            data = self.getLine()
            data[1]['meta']['top2law'] = [tobe_law[k1], tobe_law[k2]]
            if data[0] != fileIndex:
                fout.close()
                fileIndex += 1
                fout = open(outpath + str(fileIndex), 'w')
            print(json.loads(data[1]), file = fout)

        '''


    def chisquare(self):
        self.matrix = SelectKBest(chi2, k = self.dim).fit_transform(self.matrix, self.label)

    def train_test_split(self):
        self.trainx = self.matrix[:self.trainSize]
        self.testx = self.matrix[self.trainSize:]
        self.trainy = self.label[:self.trainSize]
        self.testy = self.label[self.trainSize:]
        self.matrix = []

    def quality(self, y_predict, y_test):
        microPre = metrics.precision_score(y_test, y_predict, average='micro')
        macroPre = metrics.precision_score(y_test, y_predict, average='macro')
        microRecall = metrics.recall_score(y_test, y_predict, average='micro')
        macroRecall = metrics.recall_score(y_test, y_predict, average='macro')
        f1 = metrics.f1_score(y_test, y_predict, average='weighted')
        ans = {}
        ans['microPre'] = microPre
        ans['macroPre'] = macroPre
        ans['microRecall'] = microRecall
        ans['macroRecall'] = macroRecall
        ans['F1'] = f1
        print(classification_report(y_test, y_predict, target_names=['0', '1']))

        correctFor1 = 0
        correctFor0 = 0
        falseFor1 = 0
        falseFor0 = 0
        sumFor1 = 0
        sumFor0 = 0
        for i, v in enumerate(y_test):
            if v == 1:
                sumFor1 += 1
                if y_predict[i] == 1:
                    correctFor1 += 1
                else:
                    falseFor0 += 1
            if v == 0:
                sumFor0 += 1
                if y_predict[i] == 0:
                    correctFor0 += 1
                else:
                    falseFor1 += 1

        precisionFor0 = precisionFor1 = recallFor0 = recallFor1 = 0
        try:
            precisionFor0 = (correctFor0) / (correctFor0 + falseFor0)
            recallFor0 = (correctFor0) / sumFor0

            precisionFor1 = (correctFor1) / (correctFor1 + falseFor1)
            recallFor1 = (correctFor1) / sumFor1
        except Exception as err:
            pass

        ans['0'] = {'precision': precisionFor0, 'recall': recallFor0}
        ans['1'] = {'precision': precisionFor1, 'recall': recallFor1}
        print(ans)
        return ans

'''
def accusationCMP(label, num):
    return label == num

def classifierCMP(label, num):
    return (label[0] == num[0] and label[2] == num[1])
'''

def trainAccusation():
    pass

def trainArticle():
    cl = classifier()
    print('begin read Data')
    for i in range(15):
        print(inpath + str(i))
        fin = open(inpath + str(i), 'r')
        line = fin.readline()
        while line:
            line = json.loads(line)
            cl.addData(line, test=False)
            #cl.addData(line['content'], line['meta']['law'], classifierCMP, test=False)
            line = fin.readline()
        fin.close()

    for i in range(5):
        print(inpath + str(i + 15))
        fin = open(inpath + str(i + 15), 'r')
        line = fin.readline()
        while line:
            line = json.loads(line)
            cl.addData(line, test=True)
            #cl.addData(line['content'], line['meta']['law'], classifierCMP, test=True)
            line = fin.readline()
        fin.close()
    print('read Done')
    cl.train()
    #cl.test()
    cl.predict()


trainArticle()


'''
def trainArticle():
    arrayOfClassifier = []
    xf_list = []
    fin = open('law_result1.txt', 'r')
    line = fin.readline()
    while line:
        # line = json.loads(line)
        line = line.split()
        xf_list.append([int(line[0]), int(line[1])])
        line = fin.readline()

    numberOfClassifier = len(xf_list)
    print('numberOfClassifier', numberOfClassifier)
    for num in xf_list:
        arrayOfClassifier.append(classifier(num))

    for v in arrayOfClassifier:
        print('begin read Data')
        for i in range(15):
            fin = open(inpath + str(i), 'r')
            line = fin.readline()
            while line:
                line = json.loads(line)
                v.addData(line['content'], line['meta']['law'], classifierCMP, test=False)
                line = fin.readline()
            fin.close()

        for i in range(5):
            fin = open(inpath + str(i + 15), 'r')
            line = fin.readline()
            while line:
                line = json.loads(line)
                v.addData(line['content'], line['meta']['law'], classifierCMP, test=True)
                line = fin.readline()
            fin.close()
        print('read Done')
        v.train()
        v.test()
        break

trainArticle()
'''