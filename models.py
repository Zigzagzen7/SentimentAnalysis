from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression as LRC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier as MLPC
from sklearn.pipeline import Pipeline


class Classifier():
    def __init__(self, train_data, train_label):
        # self.x = TfidfVectorizer(ngram_range=(1,2), use_idf=False, input=train_data)
        self.clf.fit(self.x, train_label)

    def evaluate(self, test_data, test_label):
        predicted = self.clf.predict(test_data)
        print('Accuracy : ', accuracy_score(predicted, test_label))
        print('Classification report : \n', classification_report(predicted, test_label))
        print('\n')


class RandomForest(Classifier):
    def __init__(self, train_data, train_label):
        print('Random Forest')
        self.clf = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), use_idf=False)),
                             ('clf', rfc(n_estimators=300, random_state=42)),
                             ])

        # self.Vectorizer = TfidfVectorizer(ngram_range=(1, 2), use_idf=False)\
        # self.x = self.Vectorizer.fit(train_data)
        # self.clf = rfc(n_estimators=300, random_state=42)
        self.clf.fit(train_data, train_label)


class MLP(Classifier):
    def __init__(self, train_data, train_label):
        self.clf = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), use_idf=False)),
                             ('clf', MLPC(hidden_layer_sizes=(100,), random_state=42)),
                             ])

        self.clf.fit(train_data, train_label)



class LogisticRegression(Classifier):
    def __init__(self, train_data, train_label):
        print('Logistic Regression')
        self.clf = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 1), use_idf=True)),
                             # ('tfidf', TfidfTransformer(use_idf=True)),
                             ('clf', LRC(penalty='l2', random_state=42)),
                             ])

        self.clf.fit(train_data, train_label)



class NaiveBayes(Classifier):
    def __init__(self, train_data, train_label):
        print('Multinomial Naive Bayes')
        self.clf = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 1), use_idf=False)),
                             ('clf', MultinomialNB(alpha=0.1)),
                             ])

        self.clf.fit(train_data, train_label)



class SVM(Classifier):
    def __init__(self, train_data, train_label):
        print('SVM')

        self.clf = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), use_idf=True)),
                             ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                                                       alpha=1e-4, max_iter=100, random_state=42, tol=1e-3)),
                             ])


        self.clf.fit(train_data, train_label)

