import warnings

from sklearn.model_selection import train_test_split

from dataloader import dataload
from models import SVM, NaiveBayes, RandomForest, LogisticRegression, MLP

PATH = 'data/'
SEED = 4

if __name__ == "__main__":
    # warnings.filterwarnings("ignore", module = r'.*sklearn')
    warnings.simplefilter(action='ignore', category=FutureWarning)

    data = dataload(PATH)

    X_train, X_test, y_train, y_test = train_test_split(data['review'], data['label'], stratify=data['label'],
                                                        test_size=0.2, random_state=SEED)

    # Multi-Layer Perceptron
    mlp = MLP(X_train, y_train)
    mlp.evaluate(X_test, y_test)

    # Random Forest
    rfc = RandomForest(X_train, y_train)
    rfc.evaluate(X_test, y_test)

    # Logistic Regression
    lr = LogisticRegression(X_train, y_train)
    lr.evaluate(X_test, y_test)

    # Naive Bayes Classifier
    nbc = NaiveBayes(X_train, y_train)
    nbc.evaluate(X_test, y_test)

    # SVM
    svm = SVM(X_train, y_train)
    svm.evaluate(X_test, y_test)

    # Save Naive Bayes Model
    # nb_pickle = open(config.naive_bayes_path, 'wb')
    # pickle.dump(nb_model_nb, nb_pickle)
    # nb_pickle.close()

    # Save SVM Model
    # svm_pickle = open(config.SVM_path, 'wb')
    # pickle.dump(nb_model_nb, svm_pickle)
    # svm_pickle.close()

    # nb_model_rfc.evaluate(X_test, y_test)
    # predicted = nb_model_rfc.predict(X_test)
    # print('results nb', nb_model_nb.loss_fct(predicted, y_test))

    # print('results nb', nb_model_nb.loss_fct(predicted, y_test))
