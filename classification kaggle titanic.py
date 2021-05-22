import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

path = 'C:/Users/Val/Desktop/Titanic/train.csv'
raw_dataset = pd.read_csv(path)
dataset = raw_dataset.copy()

dataset.pop('PassengerId')
dataset.pop('Name')

sex = dataset.pop('Sex')
dataset['Male'] = (sex == 'male')*1.0
dataset['Female'] = (sex == 'female')*1.0

dataset.pop('Ticket')  # to be included later
dataset.pop('Cabin')   # to be included later

embarked = dataset.pop('Embarked')
dataset['S'] = (embarked == 'S')*1.0
dataset['C'] = (embarked == 'C')*1.0
dataset['Q'] = (embarked == 'Q')*1.0

dataset = dataset.fillna(0)

dataset.to_csv('C:/Users/Val/Desktop/Titanic/train_smaller_processed.csv')

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_stats = train_dataset.describe()
train_stats.pop('Survived')
train_stats = train_stats.transpose()

train_labels = train_dataset.pop('Survived')
test_labels = test_dataset.pop('Survived')


def norm(x):
    return (x-train_stats['mean'])/train_stats['std']


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


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


early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=0.1)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS, validation_split=0.2, verbose=0,
                    callbacks=[early_stop, PrintDot()])
print('')
hist = pd.DataFrame(history.history)
print(hist)

model.save('titanic epoch 1000.model')

test_predictions = model.predict(normed_test_data)
predictions = [round(test_predictions[i][0]) for i in range(len(test_labels))]
print(predictions)

error = predictions - test_labels
plt.hist(error, bins=25, density=True)
plt.xlabel('Prediction Error')
plt.ylabel('Count')
plt.show()

predictions = pd.DataFrame([round(test_predictions[i][0]) for i in range(len(test_labels))])
predictions.to_csv('C:/Users/Val/Desktop/Titanic/submission.csv')
