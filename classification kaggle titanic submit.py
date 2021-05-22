import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

path = 'C:/Users/Val/Desktop/Titanic/test.csv'
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

dataset.to_csv('C:/Users/Val/Desktop/Titanic/test_smaller_processed.csv')

train_stats = dataset.describe()
train_stats = train_stats.transpose()


def norm(x):
    return (x-train_stats['mean'])/train_stats['std']


normed_test_data = norm(dataset)

model = keras.models.load_model('titanic epoch 1000.model')

test_predictions = model.predict(normed_test_data)
predictions = [round(test_predictions[i][0]) for i in range(418)]
print(predictions)

predictions = pd.DataFrame([round(test_predictions[i][0]) for i in range(418)])
predictions.to_csv('C:/Users/Val/Desktop/Titanic/submission_1.csv')
