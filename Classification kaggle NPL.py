import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import feature_extraction

path = 'C:/Users/Val/Desktop/Twitter/train.csv'
raw_dataset = pd.read_csv(path)
dataset = raw_dataset.copy()

dataset.pop('id')
dataset.pop('keyword')
dataset.pop('location')

dataset.to_csv('C:/Users/Val/Desktop/Twitter/train_processed.csv')

count_vectorizer = feature_extraction.text.CountVectorizer()
# x_train = dataset.sample(frac=0.8, random_state=0)
x_train = dataset
x_test = pd.read_csv('C:/Users/Val/Desktop/Twitter/test.csv')
x_test.pop('id')
x_test.pop('keyword')
x_test.pop('location')
x_test.to_csv('C:/Users/Val/Desktop/Twitter/test_processed.csv')
y_train = x_train.pop('target')
# y_test = x_test.pop('target')

train_vectors = count_vectorizer.fit_transform(x_train["text"])
print(x_train["text"])
print(train_vectors)
x_train_input = train_vectors.toarray()
print(x_train_input)
test_vectors = count_vectorizer.transform(x_test["text"])
x_test_input = test_vectors.toarray()
print(x_test_input)


def build_model():
    model = keras.Sequential([
        layers.Flatten(),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1, activation='sigmoid')
    ])

    LR = 0.00001
    optimizer = tf.keras.optimizers.Adam(LR)

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


model = build_model()


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')


EPOCHS = 1000


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(hist['epoch'], hist['loss'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_loss'], label='Val Error')
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(hist['epoch'], hist['accuracy'], label='Train Accuracy')
    plt.plot(hist['epoch'], hist['val_accuracy'], label='Val Accuracy')
    plt.legend()


early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(x_train_input, y_train, epochs=EPOCHS, validation_split=0.2, callbacks=[early_stop, PrintDot()],
                    verbose=0)
print("")
plot_history(history)
plt.show()

model.save('npl_full.model')

test_predictions = model.predict(x_test_input)
print(test_predictions)
predictions = [round(test_predictions.reshape((1, -1))[0][i]) for i in range(3263)]
print([round(test_predictions.reshape((1, -1))[0][i]) for i in range(3263)])
predictions = pd.DataFrame(predictions)
predictions.to_csv('C:/Users/Val/Desktop/Twitter/submission.csv')
