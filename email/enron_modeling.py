
import pickle
import numpy

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif

def preprocess(
    words_file = "./data/word_data.pkl", 
    authors_file="./data/email_authors.pkl"):
    """ 
    this function takes a pre-made list of email texts (by default word_data.pkl)
    and the corresponding authors (by default email_authors.pkl) and performs
    a number of preprocessing steps:
        -- splits into training/testing sets (10% testing)
        -- vectorizes into tfidf matrix
        -- selects/keeps most helpful features
    after this, the feaures and labels are put into numpy arrays, which play nice with sklearn functions
    4 objects are returned:
        -- training/testing features
        -- training/testing labels
    """

    # the words (features) and authors (labels), already largely preprocessed
    # this preprocessing will be repeated in the text learning mini-project
    authors_file_handler = open(authors_file, "rb")
    authors = pickle.load(authors_file_handler)
    authors_file_handler.close()

    words_file_handler = open(words_file, "rb")
    word_data = pickle.load(words_file_handler)
    words_file_handler.close()

    # test_size is the percentage of events assigned to the test set
    # (remainder go into training)
    features_train, features_test, labels_train, labels_test = train_test_split(
        word_data, 
        authors, 
        test_size = 0.1, 
        random_state = 42)

    # text vectorization--go from strings to lists of numbers
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
    features_train_transformed = vectorizer.fit_transform(features_train)
    features_test_transformed  = vectorizer.transform(features_test)



    # feature selection, because text is super high dimensional and 
    # can be really computationally chewy as a result
    selector = SelectPercentile(f_classif, percentile=10)
    selector.fit(features_train_transformed, labels_train)
    features_train_transformed = selector.transform(features_train_transformed).toarray()
    features_test_transformed  = selector.transform(features_test_transformed).toarray()

    # info on the data
    print(f"Number of Chris training emails: {sum(labels_train)}")
    print(f"Number of Sara training emails: {len(labels_train)-sum(labels_train)}")
    
    return features_train_transformed, features_test_transformed, labels_train, labels_test

features_train, features_test, labels_train, labels_test = preprocess(
    words_file = "./data/word_data.pkl", 
    authors_file="./data/email_authors.pkl")

# Train a SVM model with linear kernel
from sklearn import svm
from time import time
clf = svm.LinearSVC()

t0 = time()
model_trained = clf.fit(features_train, labels_train)
# print (f"training time: {round(time()-t0, 3)} s")

t1 = time()
labels_pred = model_trained.predict(features_test)
# print (f"prediction time: {round(time()-t1, 3)} s")
print("Number of mislabeled points out of a total %d points : %d"
    % (features_test.shape[0], (labels_test != labels_pred).sum()))

accuracy = (features_test.shape[0] - (labels_test != labels_pred).sum()) / features_test.shape[0]
print (f"accuracy: {accuracy}")

features_train.shape

features_train_small = features_train[:int(features_train.shape[0]/100)]
labels_train_small = labels_train[:int(len(labels_train)/100)]

clf_small_training_set = svm.LinearSVC()

t0 = time()
model_trained_small = clf_small_training_set.fit(features_train_small, 
    labels_train_small)
# print (f"training time: {round(time()-t0, 3)} s")

t1 = time()
labels_pred = model_trained_small.predict(features_test)
# print (f"prediction time: {round(time()-t1, 3)} s")
print("Number of mislabeled points out of a total %d points : %d"
    % (features_test.shape[0], (labels_test != labels_pred).sum()))

accuracy = (features_test.shape[0] - (labels_test != labels_pred).sum()) / features_test.shape[0]
print (f"accuracy: {accuracy}")
# =============================================================================
# SVM
# =============================================================================
clf_rbf = svm.SVC(kernel='rbf')

t0 = time()
model_trained_rbf = clf_rbf.fit(
    features_train_small, 
    labels_train_small)
# print (f"training time: {round(time()-t0, 3)} s")

t1 = time()
labels_pred = model_trained_rbf.predict(features_test)
# print (f"prediction time: {round(time()-t1, 3)} s")
print("Number of mislabeled points out of a total %d points : %d"
    % (features_test.shape[0], (labels_test != labels_pred).sum()))

accuracy = (features_test.shape[0] - (labels_test != labels_pred).sum()) / features_test.shape[0]
print (f"accuracy: {accuracy}")

# Grid search with different C values to check accuracy

from sklearn.model_selection import GridSearchCV

C_range = [10, 100, 1000., 10000.]
parameters = {'kernel':('rbf',), 'C':C_range}
svc = svm.SVC()
grid = GridSearchCV(svc, parameters)
grid.fit(
    features_train_small, 
    labels_train_small)

print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))


# =============================================================================
# naive_bayes
# =============================================================================
from sklearn.naive_bayes import GaussianNB
from time import time
gnb = GaussianNB()

t0 = time()
model_trained = gnb.fit(features_train, labels_train)
# print(f"training time: {round(time()-t0, 3)} s")

t1 = time()
labels_pred = model_trained.predict(features_test)
print(f"prediction time: {round(time()-t1, 3)} s")
print("Number of mislabeled points out of a total %d points : %d"
    % (features_test.shape[0], (labels_test != labels_pred).sum()))

accuracy = (features_test.shape[0] - (labels_test != labels_pred).sum()) / features_test.shape[0]
print(f"accuracy: {accuracy}")