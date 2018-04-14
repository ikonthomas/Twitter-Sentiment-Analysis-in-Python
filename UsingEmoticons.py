WQU Project
$This is an implementation of a tweet sentiment classifier. The accuracy on the test data containing positive and negative sentiment tweets is 84%. 
Training and test data was downloaded from Univesity of Stanford data repository


%matplotlib inline
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline

<Read data>

columns = ['polarity', 'tweetid', 'date', 'query_name', 'user', 'text']
dftrain = pd.read_csv('stanford-sentiment-twitter-data/training.1600000.processed.noemoticon.csv',
                      header = None,
                      encoding ='ISO-8859-1')
dftest = pd.read_csv('stanford-sentiment-twitter-data/testdata.manual.2009.06.14.csv',
                     header = None,
                     encoding ='ISO-8859-1')
dftrain.columns = columns
dftest.columns = columns
Text pre-processing

class RegexPreprocess(object):
    """Create a preprocessing module for a tweet or data structure of tweets.
    1) replace username, e.g., @crawles -> USERNAME
    2) replace http links -> URL
    3) replace repeated letters to two letters
    """
    
    user_pat = '(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9]+)'
    http_pat = '(https?:\/\/(?:www\.|(?!www))[^\s\.]+\.[^\s]{2,}|www\.[^\s]+\.[^\s]{2,})'
    repeat_pat, repeat_repl = "(.)\\1\\1+",'\\1\\1'
    
    def __init__(self):
        pass
    
    def transform(self, X):
        is_pd_series = isinstance(X, pd.core.frame.Series)
        if not is_pd_series:
            pp_text = pd.Series(X)
        else:
            pp_text = X
        pp_text = pp_text.str.replace(pat = self.user_pat, repl = 'USERNAME')
        pp_text = pp_text.str.replace(pat = self.http_pat, repl = 'URL')
        pp_text.str.replace(pat = self.repeat_pat, repl = self.repeat_repl)
        return pp_text
        
    def fit(self, X, y=None):
        return self
Train and test model
In [4]:
sentiment_lr = Pipeline([('regex_preprocess', RegexPreprocess()),
                         ('count_vect', CountVectorizer(min_df = 100,
                                                        ngram_range = (1,1),
                                                        stop_words = 'english')), 
                         ('lr', LogisticRegression())])
sentiment_lr.fit(dftrain.text, dftrain.polarity)
Out[4]:
Pipeline(steps=[('regex_preprocess', <__main__.RegexPreprocess object at 0x1354bc6a0>), ('count_vect', CountVectorizer(analyzer='word', binary=False, decode_error='strict',
        dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
        lowercase=True, max_df=1.0, max_features=None, min_df=10...ty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False))])
In [16]:
Xtest, ytest = dftest.text[dftest.polarity!=2], dftest.polarity[dftest.polarity!=2]
print(classification_report(ytest,sentiment_lr.predict(Xtest)))
             precision    recall  f1-score   support

          0       0.86      0.81      0.83       177
          4       0.82      0.87      0.85       182

avg / total       0.84      0.84      0.84       359

Export model for production
In [13]:
import dill
f = open('twitter_sentiment_model.pkl','wb')
r = RegexPreprocess()
dill.dump(sentiment_lr, f)
f.close()
In [15]:
# test
f = open('twitter_sentiment_model.pkl','rb')
cl = dill.load(f)
print(classification_report(ytest,cl.predict(Xtest)))
print(cl.predict_proba("Hello big beautiful world"))
f.close()
             precision    recall  f1-score   support

          0       0.86      0.81      0.83       177
          4       0.82      0.87      0.85       182

avg / total       0.84      0.84      0.84       359

[[ 0.07068138  0.92931862]]
