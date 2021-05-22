from PIL import Image
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import mnist, pandas as pd
from tensorflow.keras import layers, Sequential, callbacks, optimizers, models
import matplotlib.pyplot as plt

# x_train = mnist.train_images()
# y_train = mnist.train_labels()
# x_test = mnist.test_images()
# y_test = mnist.test_labels()
x_train = pd.read_csv('C:/Users/Val/Desktop/mnist/train.csv')
y_train = x_train.pop('label')
x_train = x_train.values
x_test = pd.read_csv('C:/Users/Val/Desktop/mnist/test.csv')

x_test = x_test.values
x_train = x_train/256
x_test = x_test/256


# clf = MLPClassifier(solver='adam', activation='relu', hidden_layer_sizes=(64, 64))
# clf.fit(x_train, y_train)
# prediction = clf.predict(x_test)
# prediction = pd.DataFrame(prediction)
# prediction.to_csv('C:/Users/Val/Desktop/mnist/submission dnn sklearn.csv')
# accuracy_1 = confusion_matrix(y_test, prediction).trace()/confusion_matrix(y_test, prediction).sum()


def build_model():
    dnn = Sequential([
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    dnn.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(0.0001))
    dnn.fit(x_train, y_train, epochs=100, validation_split=0.2, verbose=0,
            callbacks=[callbacks.EarlyStopping(patience=10, monitor='val_loss')])


dnn = models.load_model('mnist 9.model')
predictions = dnn.predict(x_test)
predictions = pd.DataFrame(np.array([np.argmax(predictions[i]) for i in range(len(predictions))]))
predictions.to_csv('C:/Users/Val/Desktop/mnist/submission dnn keras.csv')

