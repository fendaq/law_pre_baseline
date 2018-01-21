import thulac
import json
import os
from scipy.stats import chisquare
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from sklearn.externals import joblib
from gensim import corpora


inpath = '/disk/mysql/law_data/final_data/'

corpus = []

def get_corpus():
    print('reading...')
    fileList = os.listdir(inpath)
    for file in fileList:
        fin = open(inpath + file, 'r')
        line = fin.readline()
        while line:
            line = json.loads(line)
            corpus.append(line['content'].split())
            line = fin.readline()
        fin.close()
    print('finish reading')

#tfidf = TFIDF(min_df=2, strip_accents="unicode", analyzer="word", token_pattern=r"\w{1,}", ngram_range=(1,3), use_idf=1,smooth_idf=1,sublinear_tf=1)
get_corpus()
dictionary = corpora.Dictionary(corpus)
dictionary.save('dict.dict')
corpus = [dictionary.doc2bow(text) for text in corpus]
corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus)

#tfidf.fit(corpus)
#joblib.dump(tfidf, 'tfidf.model')

