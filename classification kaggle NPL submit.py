import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import feature_extraction

path = 'C:/Users/Val/Desktop/Twitter/test.csv'
raw_dataset = pd.read_csv(path)
dataset = raw_dataset.copy()

dataset.pop('id')
dataset.pop('keyword')
dataset.pop('location')

dataset.to_csv('C:/Users/Val/Desktop/Twitter/test_processed.csv')

count_vectorizer = feature_extraction.text.CountVectorizer()
x_test = dataset

count_vectorizer.fit_transform(x_test["text"])
test_vectors = count_vectorizer.transform(x_test["text"])
x_test_input = test_vectors.toarray()
print(x_test_input)

model = keras.models.load_model('npl.model')

test_predictions = model.predict(x_test_input)
predictions = [round(test_predictions.reshape((1, -1))[0][i]) for i in range(3263)]
print([round(test_predictions.reshape((1, -1))[0][i]) for i in range(3263)])
# print(y_test)

# error = predictions - y_test
# plt.hist(error, bins=25, density=True)
# plt.xlabel('Prediction Error')
# plt.ylabel('Count')
# plt.show()

predictions = pd.DataFrame(predictions)
predictions.to_csv('C:/Users/Val/Desktop/Twitter/submission.csv')
