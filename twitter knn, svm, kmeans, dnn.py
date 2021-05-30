import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import neighbors, svm, neural_network, cluster, feature_extraction
from statistics import mean


def find_x(path):
    data = pd.read_csv(path)
    x_text = data['text']
    # print(x_text)
    X_train = count_vectorizer.fit_transform(pd.read_csv(train)['text'])
    X_test = count_vectorizer.transform(x_text)
    if path == train:
        return X_train
    else:
        return X_test


def knn_acc(x_train, x_test, y_train, y_test):
    accuracy_list = []
    n_list = []
    for n in range(175):
        print("knn", n)
        knn = neighbors.KNeighborsClassifier(n_neighbors=n + 1, weights='uniform')
        knn.fit(x_train, np.ravel(y_train))
        prediction_knn = knn.predict(x_test)
        accuracy = accuracy_score(y_test, prediction_knn)
        accuracy_list += [accuracy]
        n_list += [n]
    return max(accuracy_list), n_list[accuracy_list.index(max(accuracy_list))]+1


def svm_acc(x_train, x_test, y_train, y_test):
    support = svm.SVC()
    support.fit(x_train, np.ravel(y_train))
    prediction_svm = support.predict(x_test)
    accuracy = accuracy_score(y_test, prediction_svm)
    return accuracy


def dnn_sk_acc(x_train, x_test, y_train, y_test):
    # x_train = scale(x_train)
    # x_test = scale(x_test)
    accuracy_list = []
    n_list = []
    for n in np.arange(0.0006875, 0.00072, 0.001):
        dnn = neural_network.MLPClassifier(solver='adam', activation='relu', hidden_layer_sizes=(64, 64), max_iter=1000,
                                           early_stopping=True, validation_fraction=0.2, learning_rate_init=n)
        dnn.fit(x_train, np.ravel(y_train))
        prediction_dnn = dnn.predict(x_test)
        accuracy = accuracy_score(y_test, prediction_dnn)
        accuracy_list += [accuracy]
        n_list += [n]
        print("dnn", n, accuracy)
    # plt.scatter(n_list, accuracy_list)
    # plt.show()
    return max(accuracy_list), n_list[accuracy_list.index(max(accuracy_list))]


def kmeans_acc(x_train, x_test, y_train, y_test):
    kmeans = cluster.KMeans(n_clusters=2, random_state=0)
    kmeans.fit(x_train, np.ravel(y_train))
    prediction_kmeans = kmeans.predict(x_test)
    accuracy = accuracy_score(y_test, prediction_kmeans)
    return accuracy


def init_search():
    x = find_x(train)
    y = pd.read_csv(train)[['target']].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2)

    # accuracy_knn, n_knn = knn_acc(x_train, x_test, y_train, y_test)
    accuracy_svm = svm_acc(x_train, x_test, y_train, y_test)
    accuracy_dnn_sk, n_dnn = dnn_sk_acc(x_train, x_test, y_train, y_test)
    # accuracy_kmeans = kmeans_acc(x_train, x_test, y_train, y_test)

    # print("knn: ", accuracy_knn, n_knn)
    print("svm: ", accuracy_svm)
    print("dnn sk: ", accuracy_dnn_sk, n_dnn)
    # print("kmeans: ", accuracy_kmeans)
    n_knn = 3
    return n_knn, n_dnn


def predictions(path_train, path_test):
    n_knn, n_dnn = init_search()
    x_train = find_x(path_train)
    y_train = pd.read_csv(path_train)[['target']].values
    x_test = find_x(path_test)

    # knn = neighbors.KNeighborsClassifier(n_neighbors=n_knn, weights='uniform')
    # knn.fit(x_train, np.ravel(y_train))
    # prediction_knn = knn.predict(x_test)

    # support = svm.SVC()
    # support.fit(x_train, np.ravel(y_train))
    # prediction_svm = support.predict(x_test)

    # x_train = scale(x_train)
    # x_test = scale(x_test)
    dnn = neural_network.MLPClassifier(solver='adam', activation='relu', hidden_layer_sizes=(64, 64), max_iter=1000,
                                       early_stopping=True, validation_fraction=0.2, learning_rate_init=n_dnn)
    dnn.fit(x_train, np.ravel(y_train))
    prediction_dnn = dnn.predict(x_test)

    # return [prediction_knn, prediction_svm, prediction_dnn]
    return prediction_dnn


def submit_dnn(j):
    # pred_list = predictions(train, test)
    pred_dnn = predictions(train, test)
    submit = 'C:/Users/Val/Desktop/Twitter/submission sklearn dnn ' + str(j) + '.csv'
    # prediction = pd.DataFrame(pred_list[2])
    prediction = pd.DataFrame(pred_dnn)
    prediction.columns = ['target']
    prediction.to_csv(submit)


count_vectorizer = feature_extraction.text.CountVectorizer()
train = 'C:/Users/Val/Desktop/Twitter/train.csv'
test = 'C:/Users/Val/Desktop/Twitter/test.csv'

# x = find_x(train)
# print(x)
# init_search()
submit_dnn(8)
