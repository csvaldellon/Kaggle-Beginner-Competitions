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

path = 'C:/Users/Val/Desktop/Kaggle/train_smaller_4.csv'
raw_dataset = pd.read_csv(path)
dataset = raw_dataset.copy()

paveddrive = dataset.pop('PavedDrive')
dataset['Paved'] = (paveddrive == 'Y')*1.0
dataset['Partial'] = (paveddrive == 'P')*1.0
dataset['Dirt/Gravel'] = (paveddrive == 'N')*1.0

dataset['PoolQC'] = (dataset['PoolQC'] == 'Fa')*1.0 + \
                    (dataset['PoolQC'] == 'TA')*2.0 + (dataset['PoolQC'] == 'Gd')*3.0 + (dataset['PoolQC'] == 'Ex')*4.0

dataset['Fence'] = (dataset['Fence'] == 'MnWw')*1.0 + \
                   (dataset['Fence'] == 'GdWo')*2.0 + (dataset['Fence'] == 'MnPrv')*3.0 + \
                   (dataset['Fence'] == 'GoodPrv')*4.0


dataset['MasVnrArea'] = dataset['MasVnrArea'].fillna(0)
dataset['GarageYrBlt'] = dataset['GarageYrBlt'].fillna(0)
dataset.to_csv('C:/Users/Val/Desktop/train_smaller_processed_4.csv')

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_stats = train_dataset.describe()
train_stats.pop('SalePrice')
train_stats = train_stats.transpose()

train_labels = train_dataset.pop('SalePrice')
test_labels = test_dataset.pop('SalePrice')


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


early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS, validation_split=0.2, verbose=0,
                    callbacks=[early_stop, PrintDot()]
                    )
print('')
plot_history(history)
plt.show()

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)
print("Testing set Mean Abs Error: {:5.0f} USD".format(mae))

test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [USD]')
plt.ylabel('Predictions [USD]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
plt.plot([0, 600000], [0, 600000])
plt.show()

error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [USD]')
plt.ylabel('Count')
plt.show()
