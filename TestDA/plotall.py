import os
import sys
import csv
import keras
from keras import backend as K
from keras.datasets import cifar10
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from keras.callbacks import ModelCheckpoint, Callback
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
images = []
labels = []
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall))
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
def fileReader():
    with open("files3.csv") as filename:
        template=csv.reader(filename)
        for row in template:
            rowData = []
            columnC=0
            for column in row:
                if(columnC==0):
                    image_shape = (120,120,1)
                    batch_x = np.zeros((1,) + image_shape, dtype=K.floatx())
                    img = load_img(column, target_size=(128,128), grayscale=True)
                    x = img_to_array(img)
                    images.append(x)
                if(columnC==3)
                    labels.append(column)


fileReader()
output = []
model = load_model('saved_models/23-BN-32-01UBCNN_Calcio_trained_model.h5', custom_objects={'f1': f1,'precision': precision,'recall': recall})
basePath = '34_PDC4MOHG/'

for image in images:
    batch_x[0] = image
    batch_x = batch_x/255.0

    res = model.predict_classes(batch_x)
    preout = []
    preout.append(filename)
    preout.append(res[0][0])
    output.append(preout)

with open("predictions_34_PDC4MOHG.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(output)
print(output)
