import pandas as pd, numpy as np
from sklearn.preprocessing import LabelEncoder, scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import neighbors, svm, neural_network, cluster
from statistics import mean


def find_x(path):
    data = pd.read_csv(path)

    x_categorical = data[['Sex', 'Embarked']]
    x_categorical = x_categorical.apply(lambda z: z.fillna(z.value_counts().index[0]))
    x_categorical = x_categorical.values
    Le = LabelEncoder()
    for i in range(len(x_categorical[0])):
        x_categorical[:, i] = Le.fit_transform(x_categorical[:, i])
    x_categorical = pd.DataFrame(x_categorical)

    x_quantitative = data[['Pclass', 'SibSp', 'Parch', 'Fare', 'Age']]
    x_quantitative = x_quantitative.fillna(x_quantitative.mean())

    x = x_categorical.join(x_quantitative)
    return x


def knn_acc(x_train, x_test, y_train, y_test):
    accuracy_list = []
    n_list = []
    for n in range(175):
        # print("knn", n)
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
    x_train = scale(x_train)
    x_test = scale(x_test)
    accuracy_list = []
    n_list = []
    for n in np.arange(0.0001, 0.0002, 0.0001):
        # print("dnn", n)
        dnn = neural_network.MLPClassifier(solver='adam', activation='relu', hidden_layer_sizes=(64, 64), max_iter=1000,
                                           early_stopping=True, validation_fraction=0.2, learning_rate_init=n)
        dnn.fit(x_train, np.ravel(y_train))
        prediction_dnn = dnn.predict(x_test)
        accuracy = accuracy_score(y_test, prediction_dnn)
        accuracy_list += [accuracy]
        n_list += [n]
    return max(accuracy_list), n_list[accuracy_list.index(max(accuracy_list))]


def kmeans_acc(x_train, x_test, y_train, y_test):
    kmeans = cluster.KMeans(n_clusters=2, random_state=0)
    kmeans.fit(x_train, np.ravel(y_train))
    prediction_kmeans = kmeans.predict(x_test)
    accuracy = accuracy_score(y_test, prediction_kmeans)
    return accuracy


def column_finder(acc_tol):
    features = find_x(train)
    column_knn, column_svm, column_dnn, column_kmeans = [], [], [], []
    column_filtered = [column_knn, column_svm, column_dnn, column_kmeans]
    for column in features.keys():
        x = features[[column]].values
        y = pd.read_csv(train)[['Survived']].values
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        accuracy_knn, n_knn = knn_acc(x_train, x_test, y_train, y_test)
        accuracy_svm = svm_acc(x_train, x_test, y_train, y_test)
        accuracy_dnn_sk, n_dnn = dnn_sk_acc(x_train, x_test, y_train, y_test)
        accuracy_kmeans = kmeans_acc(x_train, x_test, y_train, y_test)

        acc_list = [accuracy_knn, accuracy_svm, accuracy_dnn_sk, accuracy_kmeans]
        i = 0
        for acc in acc_list:
            if acc >= acc_tol:
                column_filtered[i] += [column]
            i += 1
    return column_knn, column_svm, column_dnn, column_kmeans


def init_search():
    x = find_x(train)
    y = pd.read_csv(train)[['Survived']].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2)

    accuracy_knn, n_knn = knn_acc(x_train, x_test, y_train, y_test)
    accuracy_svm = svm_acc(x_train, x_test, y_train, y_test)
    accuracy_dnn_sk, n_dnn = dnn_sk_acc(x_train, x_test, y_train, y_test)
    accuracy_kmeans = kmeans_acc(x_train, x_test, y_train, y_test)

    print("knn: ", accuracy_knn, n_knn)
    # print("svm: ", accuracy_svm)
    # print("dnn sk: ", accuracy_dnn_sk, n_dnn)
    # print("kmeans: ", accuracy_kmeans)

    return n_knn, n_dnn


def init_avg_search(max_iter):
    x = find_x(train)
    y = pd.read_csv(train)[['Survived']].values
    list_knn, list_n_knn, list_svm, list_dnn_sk, list_n_dnn, list_kmeans = [], [], [], [], [], []
    for m in range(max_iter):
        print(m)
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2)

        accuracy_knn, n_knn = knn_acc(x_train, x_test, y_train, y_test)
        accuracy_svm = svm_acc(x_train, x_test, y_train, y_test)
        accuracy_dnn_sk, n_dnn = dnn_sk_acc(x_train, x_test, y_train, y_test)
        accuracy_kmeans = kmeans_acc(x_train, x_test, y_train, y_test)

        list_knn += [accuracy_knn]
        list_n_knn += [n_knn]
        list_svm += [accuracy_svm]
        list_dnn_sk += [accuracy_dnn_sk]
        list_n_dnn += [n_dnn]
        list_kmeans += [accuracy_kmeans]

    return mean(list_knn), mean(list_n_knn), mean(list_svm), mean(list_dnn_sk), mean(list_n_dnn), mean(list_kmeans)


def predictions(path_train, path_test):
    n_knn, n_dnn = init_search()
    x_train = find_x(path_train)
    y_train = pd.read_csv(path_train)[['Survived']].values
    x_test = find_x(path_test)

    knn = neighbors.KNeighborsClassifier(n_neighbors=n_knn, weights='uniform')
    knn.fit(x_train, np.ravel(y_train))
    prediction_knn = knn.predict(x_test)

    support = svm.SVC()
    support.fit(x_train, np.ravel(y_train))
    prediction_svm = support.predict(x_test)

    x_train = scale(x_train)
    x_test = scale(x_test)
    dnn = neural_network.MLPClassifier(solver='adam', activation='relu', hidden_layer_sizes=(64, 64), max_iter=1000,
                                       early_stopping=True, validation_fraction=0.2, learning_rate_init=n_dnn)
    dnn.fit(x_train, np.ravel(y_train))
    prediction_dnn = dnn.predict(x_test)

    return [prediction_knn, prediction_svm, prediction_dnn]


def submission():
    j = 1
    for prediction in predictions(train, test):
        submit = 'C:/Users/Val/Desktop/Titanic/submission sklearn ' + str(j) + '.csv'
        prediction = pd.DataFrame(prediction)
        prediction.columns = ['Survived']
        prediction.to_csv(submit)
        j += 1


def submit_dnn(j):
    pred_list = predictions(train, test)
    submit = 'C:/Users/Val/Desktop/Titanic/submission sklearn dnn ' + str(j) + '.csv'
    prediction = pd.DataFrame(pred_list[2])
    prediction.columns = ['Survived']
    prediction.to_csv(submit)


train = 'C:/Users/Val/Desktop/Titanic/train.csv'
test = 'C:/Users/Val/Desktop/Titanic/test.csv'
# init_search()
# submission()
# col_knn, col_svm, col_dnn, col_kmeans = column_finder(0.3)
# print(col_knn, col_svm, col_dnn, col_kmeans)
# A, A_n, B, C, C_n, D = init_avg_search(max_iter=30)
# print(C, C_n)

# for k in range(5):
# submit_dnn(5)
for n in range(10):
    init_search()
