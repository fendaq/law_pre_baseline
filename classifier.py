import json
import os
from scipy.stats import chisquare
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from sklearn.externals import joblib
from gensim import corpora
from gensim.corpora.textcorpus import TextCorpus
from gensim.corpora.mmcorpus import MmCorpus
from gensim.models.tfidfmodel import TfidfModel



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
    def __init__(self, num, dim):
        self.num = num   #罪名代号
        self.accusationName = accusation_list[num] #罪名名称
        self.dim = dim  #降维之后的向量维数

    def getData(self):

        pass

    def train(self):
        self.getData()
        self.chisquare()



    def chisquare(self):
        pass

    def predict(self, text):
        pass

