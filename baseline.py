#imports
import gensim, time
from utils.dataset import DataSet
from utils.generate_test_splits import split
from utils.score import report_score
import nltk, re 
from nltk.corpus import stopwords, brown
from collections import OrderedDict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
import numpy as np
from gensim.models import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.learning_curve import learning_curve
import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score, make_scorer
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression


dataset = DataSet()
data_splits = split(dataset)

training_data = data_splits['training']
dev_data = data_splits['dev']
test_data = data_splits['test']



LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
LABELS_RELATED = ['unrelated','related']
RELATED = LABELS[0:3]


def bow(trainingData, articles):
    cachedStopWords = stopwords.words("english") 
    list_label = []
    list_of_lists = []

    for i in range(0, len(trainingData)):
        list_headlines = []
        list_articles = []
        list_combine = []
        list_tokens = []

        list_label.append(trainingData[i]['Stance'])

        list_headlines = trainingData[i]['Headline']

        for key in articles:
            if key == trainingData[i]['Body ID']:
                list_articles = articles[key]

        list_combine = list_headlines + list_articles

        tokens = nltk.word_tokenize(list_combine)
        for token in tokens:
            token = re.sub("[^A-Za-z]+", "", token)
            token = token.lower()

            if (token not in cachedStopWords):
                list_tokens.append(token)

        joined = (" ".join(list_tokens))

        list_of_lists.append(joined)

    return(list_of_lists, list_label)





def tfidf (train_data, dev_data, test_data):

    tf_train = TfidfVectorizer(analyzer = 'word', preprocessor = None, stop_words = None, max_features = 500, min_df = 0, lowercase = False, ngram_range = (1,3))

    tfidf_matrix_train = tf_train.fit_transform(train_data).toarray()
    tfidf_matrix_dev = tf_train.transform(dev_data).toarray()
    tfidf_matrix_test = tf_train.transform(test_data).toarray()

    return(tfidf_matrix_train, tfidf_matrix_dev, tfidf_matrix_test)

def classifier(train_data, labels, test_data):
    #clf = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth=3, min_samples_leaf=5)
    #clf = svm.SVC()
    #clf = GaussianNB()
    clf = LogisticRegression(C = 1.0, class_weight='balanced', solver="lbfgs", max_iter=150) 
    clf.fit(train_data, labels)

    return(clf.predict(test_data))



if __name__ == "__main__":
    
    start_time = time.time()

    bow_train, labels_train = bow(training_data, dataset.articles)
    bow_dev, labels_dev = bow(dev_data, dataset.articles)
    bow_test, labels_test = bow(test_data, dataset.articles)

    tfidf_matrix_train, tfidf_matrix_dev, tfidf_matrix_test = tfidf(bow_train, bow_dev, bow_test) 
    
    predicted = classifier(tfidf_matrix_train, labels_train, tfidf_matrix_dev)

    report_score(labels_dev, predicted)
   


    print("--- %s seconds ---" % (time.time() - start_time))





