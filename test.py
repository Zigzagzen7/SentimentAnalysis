import pickle
import warnings

from sklearn.model_selection import train_test_split

from dataloader import dataload

PATH = 'data/'
# SEED = 108
SPLIT = 0.8

if __name__ == "__main__":

    warnings.simplefilter(action='ignore', category=FutureWarning)
    data = dataload(PATH)

    pickle_in = open("models" + ".pkl", "rb")
    model = pickle.load(pickle_in)

    SEED = model[0]

    X_train, X_test, y_train, y_test = train_test_split(data['review'], data['label'], stratify=data['label'],
                                                        train_size=SPLIT, random_state=SEED)

    print("Evaluating performance of models on test set")
    for i in range(1, 6):
        model[i].evaluate(X_test, y_test)


