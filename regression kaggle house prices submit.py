import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

train_dataset = pd.read_csv('C:/Users/Val/Desktop/Kaggle/train_smaller_processed_4.csv')
test_dataset = pd.read_csv('C:/Users/Val/Desktop/test_smaller_processed.csv')

train_stats = train_dataset.describe()
train_stats.pop('SalePrice')
train_stats = train_stats.transpose()

train_labels = train_dataset.pop('SalePrice')


def norm(x):
    return (x-train_stats['mean'])/train_stats['std']


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1)
    ])

    LR = 0.001
    optimizer = tf.keras.optimizers.Adam(LR)

    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
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
    plt.ylabel('Mean Abs Error [USD]')
    plt.plot(hist['epoch'], hist['mae'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'], label='Val Error')
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$USD^2$]')
    plt.plot(hist['epoch'], hist['mse'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'], label='Val Error')
    plt.legend()


# early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS, validation_split=0.2, verbose=0,
                    callbacks=[PrintDot()]
                    )
print('')
plot_history(history)
plt.show()


test_predictions = pd.DataFrame(model.predict(normed_test_data).flatten())
test_predictions.to_csv('C:/Users/Val/Desktop/Kaggle/submission_2.csv')
