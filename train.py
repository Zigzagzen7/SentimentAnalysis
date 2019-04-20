'''
    Alexandre Marcil
    18 avril 2019
    inspired from https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
    dataset from https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences
'''


import warnings
import pickle

from sklearn.model_selection import train_test_split

from dataloader import dataload
from models import SVM, NaiveBayes, RandomForest, LogisticRegression, MLP

PATH = 'data/'
SEED = 108
SPLIT = 0.8

if __name__ == "__main__":

    warnings.simplefilter(action='ignore', category=FutureWarning)

    data = dataload(PATH)

    X_train, X_test, y_train, y_test = train_test_split(data['review'], data['label'], stratify=data['label'],
                                                        train_size=SPLIT, random_state=SEED)

    print("Training models and printing training accuracies...")
    # Random Forest
    rfc = RandomForest(X_train, y_train)
    rfc.evaluate(X_train, y_train)

    # Naive Bayes Classifier
    nbc = NaiveBayes(X_train, y_train)
    nbc.evaluate(X_train, y_train)

    # SVM
    svm = SVM(X_train, y_train)
    svm.evaluate(X_train, y_train)

    # Logistic Regression
    lrc = LogisticRegression(X_train, y_train)
    lrc.evaluate(X_train, y_train)

    # Multi-Layer Perceptron
    mlp = MLP(X_train, y_train)
    mlp.evaluate(X_train, y_train)

    # Saving our models as a dictionary with 5 components
    saved_model = open("models" + ".pkl", "wb")
    pickle.dump({0: SEED, 1: rfc, 2: nbc, 3: svm, 4: lrc, 5: mlp}, saved_model)
    saved_model.close()


