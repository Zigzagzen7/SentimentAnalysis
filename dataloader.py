import glob
import pandas as pd
import json as j
import pandas as pd
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2


def getReviews(file):
    f=open(file)
    lines=f.readlines()
    f.close()
    reviews=[]
    for l in lines:
        x=l.split('\t')
        reviews.append([str.lstrip(str.rstrip(x[0])),str.lstrip(str.rstrip(x[1]))])
    return reviews


def dataload(path):
    reviewsList = []
    for files in glob.glob(path + "/*.txt"):
        reviewsList += (getReviews(files))
    df =  pd.DataFrame(reviewsList, columns=['review', 'label'])

    df['label']= df['label'].astype(int)

    # print(df['label'][0])
    return df

