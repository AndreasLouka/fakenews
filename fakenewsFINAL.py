#imports:
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
    #Bag of words feature representation of datasets:#

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



def world_overlap(trainingData, articles):
    #world overlap features present both in headline and body text:#

    cachedStopWords = stopwords.words("english") 
    world_overlap_features = []

    for i in range(0, len(trainingData)):
        list_headlines = []
        list_tokens_headlines = []
        list_articles = []
        list_tokens_articles = []


        list_headlines = trainingData[i]['Headline']

        tokens_headlines = nltk.word_tokenize(list_headlines)
        for token_h in tokens_headlines:
            token_h = re.sub("[^A-Za-z]+", "", token_h)
            token_h = token_h.lower()

            if (token_h not in cachedStopWords):
                list_tokens_headlines.append(token_h)


        for key in articles:
            if key == trainingData[i]['Body ID']:
                list_articles = articles[key]

        tokens_articles = nltk.word_tokenize(list_articles)
        for token_a in tokens_articles:
            token_a = re.sub("[^A-Za-z]+", "", token_a)
            token_a = token_a.lower()

            if (token_a not in cachedStopWords):
                list_tokens_articles.append(token_a)

        #world overlap features:
        world_overlap = [len(set(list_tokens_headlines).intersection(list_tokens_articles)) / float(len(set(list_tokens_headlines).union(list_tokens_articles)))]
        world_overlap_features.append(world_overlap)


    return(np.asarray(world_overlap_features))



def binary_co_occurence_stops(trainingData, articles):
    # count features for tokens present both in healdine and body text:#

    cachedStopWords = stopwords.words("english") 
    co_occurance = []

    for i in range(0, len(trainingData)):
        count = 0
        count_occurance = []
        list_headlines = []
        list_tokens_headlines = []
        list_articles = []
        list_tokens_articles = []


        list_headlines = trainingData[i]['Headline']

        tokens_headlines = nltk.word_tokenize(list_headlines)
        for token_h in tokens_headlines:
            token_h = re.sub("[^A-Za-z]+", "", token_h)
            token_h = token_h.lower()

            if (token_h not in cachedStopWords):
                list_tokens_headlines.append(token_h)


        for key in articles:
            if key == trainingData[i]['Body ID']:
                list_articles = articles[key]

        tokens_articles = nltk.word_tokenize(list_articles)
        for token_a in tokens_articles:
            token_a = re.sub("[^A-Za-z]+", "", token_a)
            token_a = token_a.lower()

            if (token_a not in cachedStopWords):
                list_tokens_articles.append(token_a)

        for headline_token in list_tokens_headlines:
            if headline_token in list_tokens_articles:
                count += 1
        count_occurance.append(count)

        co_occurance.append(count_occurance)

    return(np.asarray(co_occurance))



def tfidf (train_data, dev_data, test_data):
    #tfidf matrices:#

    tf_train = TfidfVectorizer(analyzer = 'word', preprocessor = None, stop_words = None, max_features = 500, min_df = 0, lowercase = False, ngram_range = (1,3))

    tfidf_matrix_train = tf_train.fit_transform(train_data).toarray()
    tfidf_matrix_dev = tf_train.transform(dev_data).toarray()
    tfidf_matrix_test = tf_train.transform(test_data).toarray()

    return(tfidf_matrix_train, tfidf_matrix_dev, tfidf_matrix_test)



def cos_similarity(data):
    #calculation of cosine angle similarity between headlines and body texts:#

    tf = TfidfVectorizer(analyzer = 'word', preprocessor = None, stop_words = None, max_features = 500, min_df = 0, lowercase = False, ngram_range = (1,3))

    body = []
    for i in range(len(data)):
        body.append(dataset.articles[data[i]['Body ID']])

    headline = []
    for i in range(len(data)):
        headline.append(data[i]['Headline'])

    features = []
    for i in range(0, len(body)):
        the_list = []
        the_list.append(body[i])
        the_list.append(headline[i])
        tfidf = tf.fit_transform(the_list)

        similarity = (tfidf * tfidf.T).A
        features.append(similarity[0])

    return (np.asarray(features))



def read_corpus_doc2vec(f, tokens_only=False):
    #function used to create lists for doc2vec:#

        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])



def doc2vec_arrays (corpus):
    #doc2vec arrays:#

    model = Doc2Vec.load('the_model_doc2vec.doc2vec') #load model previously saved as 'doc2vev_model.doc2vec'

    vectors_list = []

    for doc_id in range(len(corpus)):
        inferred_vector = model.infer_vector(corpus[doc_id].words)
        vectors_list.append(inferred_vector)

    vectors_array = np.asarray(vectors_list)

    return(vectors_array)



def combine_featues (array1, array2, array3, array4, array5):
    #function to combine all the features extracted:#

    return(np.hstack((array1, array2, array3, array4, array5)))



def classifier(train_data, labels, test_data):
    #classifiers:#

    clf = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth=3, min_samples_leaf=5)
    #clf = GaussianNB()
    #clf = svm.SVC()
    #clf = LogisticRegression(C = 1.0, class_weight='balanced', solver="lbfgs", max_iter=150) 


    clf.fit(train_data, labels)

    return(clf.predict(test_data))



def plot_curve(X, y):
    #plotting learning curves for classifiers:#

    # ** function plot_curve modified from: http://www.ritchieng.com/machinelearning-learning-curve/ ** #

    size = len(X)
    cv = KFold(size, shuffle=True)

    #Classifiers:
    clf = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth=3, min_samples_leaf=5)
    #clf = svm.SVC()
    #clf = GaussianNB()
    #clf = LogisticRegression(C = 1.0, class_weight='balanced', solver="lbfgs", max_iter=150) 


    # fit
    clf.fit(X, y)
    
    train_sizes, train_scores, test_scores = learning_curve(clf, X, y, n_jobs=-1, cv=cv, train_sizes=np.linspace(.1, 1.0, 5), verbose=0)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.figure()
    plt.title("Decision Tree")
    plt.legend(loc="best")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.gca().invert_yaxis()
    
    # box-like grid
    plt.grid()
    
    # plot the std deviation as a transparent range at each training set size
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
    # plot the average training and test score lines at each training set size
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    
    
    plt.ylim(.0,0.9)
    plt.show()





if __name__ == "__main__":
    
    #Execution time: 
    start_time = time.time()

    ###############

    #A) VARIABLES:

        #1) bag of words features:
    bow_train, labels_train = bow(training_data, dataset.articles)
    bow_dev, labels_dev = bow(dev_data, dataset.articles)
    bow_test, labels_test = bow(test_data, dataset.articles)

        #2) world overlap features:
    train_overlap_array = world_overlap(training_data, dataset.articles)
    dev_overlap_array = world_overlap(dev_data, dataset.articles)
    test_overlap_array = world_overlap(test_data, dataset.articles)

        #3) binary co occurance features:
    train_co_occurance = binary_co_occurence_stops(training_data, dataset.articles)
    dev_co_occurance = binary_co_occurence_stops(dev_data, dataset.articles)
    test_co_occurance = binary_co_occurence_stops(test_data, dataset.articles)

        #4) tfidf features:
    tfidf_matrix_train, tfidf_matrix_dev, tfidf_matrix_test = tfidf(bow_train, bow_dev, bow_test) 

        #5) cosine similarity features:
    train_similarity_array = cos_similarity(training_data)
    dev_similarity_array = cos_similarity(dev_data)
    test_similarity_array = cos_similarity(test_data)

    #doc2vec model:
    model = gensim.models.doc2vec.Doc2Vec(size=50, min_count=2, iter=150)

        #6) Variables for doc2vec:
    train_corpus = list(read_corpus_doc2vec(bow_train)) #create corpus in the appropriate format for doc2vec
    dev_corpus = list(read_corpus_doc2vec(bow_dev))
    test_corpus = list(read_corpus_doc2vec(bow_test)) 

            #6.1) Train the doc2vec model and save it (only need to run once):
    # model.build_vocab(train_corpus)
    # model.train(train_corpus, total_examples=model.corpus_count)
    # model.save('the_model_doc2vec.doc2vec')

            #6.2) Doc2vec arrays:
    train_vectors_array = doc2vec_arrays(train_corpus)
    dev_vectors_array = doc2vec_arrays(dev_corpus)
    test_vectors_array = doc2vec_arrays(test_corpus)


        #7) Combine all features together:
    train_array = combine_featues(train_vectors_array, train_overlap_array, tfidf_matrix_train, train_similarity_array, train_co_occurance)
    dev_array = combine_featues(dev_vectors_array, dev_overlap_array, tfidf_matrix_dev, dev_similarity_array, dev_co_occurance)
    test_array = combine_featues(test_vectors_array, test_overlap_array, tfidf_matrix_test, test_similarity_array, test_co_occurance)
    
    ###############

    #B) Classifier: 
    
    predicted = classifier(train_array, labels_train, test_array)


    ###############

    #C) Scoring:

    report_score(labels_test, predicted)


    ###############

    #D) Learning Curves:
    
    #plot_curve(train_array, labels_train)


    ###############

    #E) Execution time: 
    print("--- %s seconds ---" % (time.time() - start_time))
