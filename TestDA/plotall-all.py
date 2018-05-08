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
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
images = []
labelsP = []
filenames = []
preFilen = []
filenFil = []
labelFil = []
outputAll = []
pullbacks = []
testIndex = []
istest = []
testPath = []
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
    with open("clean_labels_giraffe.csv") as filename:
        template=csv.reader(filename)
        last = ''
        add = 0
        labels = []
        coun = 0
        for row in template:
            rowData = []
            columnC=0
            for column in row:
                if(columnC==0):
                    ptmp = column.split('/')
                    if (last!=ptmp[1]):
                        last=ptmp[1]
                        if(coun!=0):
                            add = 1
                    else:
                        add = 0
                    filenames.append(column)
                if(columnC==3): 
                    if(add==1):
                        labelsP.append(labels)
                        labels = []
                    labels.append(column)
                columnC+=1
                coun+=1
    print(labelsP)
    print 'Success!!'
    print(len(labelsP[0]))
    with open("clean_labels_giraffe.csv") as filename:
        template=csv.reader(filename)
        last = ''
        for row in template:
            rowData = []
            columnC=0
            for column in row:
                if(columnC==0):
                    ptmp = column.split('/')
                    if (last!=ptmp[1]):
                        pullbacks.append(ptmp[1])
                        last=ptmp[1]
                    columnC+=1
    pullbacks.sort()
    #print pullbacks
    with open("saved_models/3_3-BN-32-01UBCNN_Calcio_trained_model.h5.csv") as filename:
        template=csv.reader(filename)
        for row in template:
            rowData = []
            columnC=0
            for column in row:
                tmp =int(column)
                if(tmp>33 and tmp<45):
                    tmp-=1
                elif (tmp>45 and tmp<71):
                    tmp-=2
                elif (tmp>71):
                    tmp-=3
                testIndex.append(tmp)
    for tsti in testIndex:
        testPath.append(pullbacks[int(tsti)])

def plotter():
    cnt = 0
    print labelsP[8]
    for pullback in outputAll:
        coun = 0
        fp = 0.0000
        fn = 0.0000
        tp = 0.0000
        tn = 0.0000
    	for xp in pullback:
    		tmp = int(labelsP[cnt][coun])
    		if(xp==0):
    			if(tmp == 0):
    				fn=fn+1
    			if(tmp==1):
    				fp=fp+1
    		if(xp==1):
    			if(tmp==0):
    				tn=tn+1
    			if(tmp==1):
    				tp=tp+1
    		coun+=1
    	print(fp,fn,tp,tn)
        if(tp!=0):
        	prec = tp/(tp+fp)
        	rec = tp/(tp+fn)
        else:
            prec = 0
            rec=0
    	print(prec,rec)
        if(tp!=0 and prec != 0 or rec!=0):
    	   f1 = 2*((prec*rec)/(prec+rec))
        else:
            f1 = 'nan'
        plt.figure(num=None, figsize=(10, 6), dpi=150)
        plt.ylim(-0.5,1.5)
    	plt.plot(labelsP[cnt], color='blue')
    	plt.plot(pullback, color='black' , linestyle='--') 
    	black_patch = mpatches.Patch(color='black', label='Network Prediction')
    	blue_patch = mpatches.Patch(color='blue', label='Original Labels')	
    	plt.legend(handles=[blue_patch,black_patch])
        plte = 0
        for tst in testIndex:
            if(int(tst)==cnt):
                plte=1
        if(plte==1):
            print 'TEEEEST ' + pullbacks[cnt]
            plt.xlabel(pullbacks[cnt] + ' TEST')
        else:
            plt.xlabel(pullbacks[cnt] + ' TRAINING')

        if(pullbacks[cnt]=="34_PDC4MOHG" or pullbacks[cnt]=="46_PD2DK5KB" or pullbacks[cnt]=="72_PD2D493T"):
            plt.xlabel(pullbacks[cnt] + ' N-1')       

    	plt.ylabel('F1: ' + str(f1) + ' Prec: ' + str(prec) + ' Rec: ' + str(rec))
    	plt.savefig(pullbacks[cnt]+'.png')
        cnt+=1
def predicter():
    model = load_model('saved_models/3_3-BN-32-01UBCNN_Calcio_trained_model.h5', custom_objects={'f1': f1,'precision': precision,'recall': recall})
    print 'loaded saved_models/3_3-BN-32-01UBCNN_Calcio_trained_model.h5 '
    last = ''
    count = 0
    output = []
    for image in filenames:
        tmpi = image.split('/')[1]
        if(tmpi!=last):
            print tmpi
            if(count!=0):
                print 'appending ' + last + '...'
                outputAll.append(output)
            output = []
            last = tmpi
        image_shape = (120,120,1)
        batch_x = np.zeros((1,) + image_shape, dtype=K.floatx())
        img = load_img(image, target_size=(120,120), grayscale=True)
        x = img_to_array(img)
        x/=255
        batch_x[0] = x

        res = model.predict_classes(batch_x)
        #print image + ' was ' + str(res[0][0])
        output.append(res[0][0])
        count+=1

fileReader()
predicter()
plotter()

