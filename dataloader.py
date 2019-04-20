import glob

import pandas as pd


def getreviews(file):
    f = open(file)
    lines = f.readlines()
    f.close()
    reviews = []
    for l in lines:
        x = l.split('\t')
        reviews.append([str.lstrip(str.rstrip(x[0])), str.lstrip(str.rstrip(x[1]))])
    return reviews


def dataload(path):
    reviews_list = []
    for files in glob.glob(path + "/*.txt"):
        reviews_list += (getreviews(files))
    df = pd.DataFrame(reviews_list, columns=['review', 'label'])
    df['label'] = df['label'].astype(int)  # for classification report

    return df

