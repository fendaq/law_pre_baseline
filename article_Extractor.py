import thulac
import json
import os
from scipy.stats import chisquare
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from sklearn.externals import joblib
from gensim import corpora
from gensim.corpora.textcorpus import TextCorpus
from gensim.corpora.mmcorpus import MmCorpus
from gensim.models.tfidfmodel import TfidfModel

#this program is used to get tfidf model

inpath = '/disk/mysql/law_data/final_data/'
modelpath = 'model/'


class law_corpus(TextCorpus):
    def get_texts(self):
        print('read to get corpus...')
        fileList = os.listdir(inpath)
        for fname in fileList:
            fin = open(inpath + fname, 'r')
            line = fin.readline()
            while line:
                line = json.loads(line)
                yield line['content'].split()
                line = fin.readline()
            fin.close()
            break
        print('finish reading...')

#used to build dictionary
class MyCorpus():
    def __iter__(self):
        print('begin iteration...')
        fileList = os.listdir(inpath)
        for fname in fileList:
            fin = open(inpath + fname, 'r')
            line = fin.readline()
            while line:
                line = json.loads(line)
                yield line['content'].split()
                line = fin.readline()
            fin.close()
            #break
        print('end iteration...')

class streamCorpus():
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __iter__(self):
        print('begin iteration...')
        fileList = os.listdir(inpath)
        for fname in fileList:
            fin = open(inpath + fname, 'r')
            line = fin.readline()
            while line:
                line = json.loads(line)
                yield self.dictionary.doc2bow(line['content'].split())
                line = fin.readline()
            fin.close()
            #break
        print('end iteration...')


def buildDictionary(corpus):
    #corpus = MyCorpus()
    print('get dictionary...')
    if not os.path.exists(modelpath + 'dictionary.model'):
        # 构造词典
        dictionary = corpora.Dictionary(corpus)
        print(dictionary)
        dictionary.save(modelpath + 'dictionary.model')
    else:
        dictionary = corpora.Dictionary.load(modelpath + 'dictionary.model')
    print('done')

    return dictionary


def buildTfidfModel(corpus):
    print('get tfidf model...')
    if not os.path.exists(modelpath + 'tfidf.model'):
        # 构造tfidf向量
        tfidf = TfidfModel(corpus)
        tfidf.save(modelpath + 'tfidf.model')
    else:
        tfidf = TfidfModel.load(modelpath + 'tfidf.model')
    print('done')
    return tfidf


def build():
    print('begin build tfidf model ...')

    corpus = MyCorpus()
    dictionary = buildDictionary(corpus)
    corpus = streamCorpus(dictionary)
    tfidf = buildTfidfModel(corpus)


def getLabel():
    label = []
    fileList = os.listdir(inpath)
    for fname in fileList:
        fin = open(inpath + fname, 'r')
        line = fin.readline()
        while line:
            line = json.loads(line)
            label.append(line['meta'][''])
            line = fin.readline()
        fin.close()

build()

#scipy_csc_matrix = gensim.matutils.corpus2csc(corpus)
#↑把语料库转换成稀疏矩阵


from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2




