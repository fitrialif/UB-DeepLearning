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
from skimage.transform import resize

images = []
labelsP = []
filenames = []
preFilen = []
filenFil = []
labelFil = []
outputAll = []
pullbacks = []
testIndex = []
trainIndex = []
istest = []
testPath = []
trainPath = []
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
                    labels.append(int(column))
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
    #with open("saved_models/1_TEST_4-BN-32-01UBCNN_Calcio_trained_model.h5.csv") as filename:
    with open("saved_models/0_TEST_4-BN-32-01UBCNN_Calcio_trained_model.h5.csv") as filename:
        template=csv.reader(filename)
        for row in template:
            rowData = []
            columnC=0
            for column in row:
                tmp =int(column)
                testIndex.append(tmp)
    for tsti in testIndex:
        testPath.append(pullbacks[int(tsti)])
        #print pullbacks
    #with open("saved_models/1_TRAIN_4-BN-32-01UBCNN_Calcio_trained_model.h5.csv") as filename:
    with open("saved_models/0_TRAIN_4-BN-32-01UBCNN_Calcio_trained_model.h5.csv") as filename:
        template=csv.reader(filename)
        for row in template:
            rowData = []
            columnC=0
            for column in row:
                tmp =int(column)
                trainIndex.append(tmp)
    for tsti in trainIndex:
        trainPath.append(pullbacks[int(tsti)])

def plotter():
    cnt_all = 0
    cnt_train = 0
    cnt_extrain = 0
    prec_all = 0
    prec_train = 0
    prec_extrain = 0
    rec_all = 0
    rec_train = 0
    rec_extrain = 0
    acc_all = 0
    acc_train = 0
    acc_extrain = 0
    cnt = 0
    sc = 0
    sctr = 0
    scts = 0
    scex = 0
    print labelsP[8]
    for pullback in outputAll:
        basetitle = '4-BN-32-01'
        coun = 0
        fp = 0.0000
        fn = 0.0000
        tp = 0.0000
        tn = 0.0000
        prec = 0.0000
        rec = 0.0000
        #Calculo tp tn fp fn
    	for xp in pullback:
            #Lector de output labels original
    		tmp = int(labelsP[cnt][coun])
    		if(xp==0):
    			if(tmp == 0):
    				tn=tn+1
    			if(tmp==1):
    				fn=fn+1
    		if(xp==1):
    			if(tmp==0):
    				fp=fp+1
    			if(tmp==1):
    				tp=tp+1
    		coun+=1
        print(pullbacks[cnt])
    	print(fp,fn,tp,tn)
        if((tp+fp)==0):
            prec = 1
            rec = 0
            if((tp+fn)==0):
                rec = 1
        elif((tp+fn)==0):
            rec = 1
            prec = 0
            if((tp+fp)==0):
                prec = 1
        else:
            prec = tp/(tp+fp)
            rec = tp/(tp+fn)
        acc = (tp+tn)/len(pullback)
        print(prec,rec,acc)
        if(prec==0 and rec==0):
            f1=0
        else:
            f1 = 2*((prec*rec)/(prec+rec))
        if(f1==0):
            print('Badddddddddddddddddddddddddddddddddddddddddddddddddddddddd')
        plt.figure(num=None, figsize=(10, 6), dpi=150)
    	axes = plt.gca()
        print str(len(pullback)) + '=' + str(len(labelsP[cnt]))
        plt.plot(labelsP[cnt], color='black', linestyle=':', marker='.')
    	plt.plot(pullback, color='red' , linestyle=':', marker='x',alpha=0.5) 
        plt.ylim(-0.5,1.5)
    	black_patch = mpatches.Patch(color='red', label='Network Prediction')
    	blue_patch = mpatches.Patch(color='black', label='Original Labels')	
    	plt.legend(handles=[blue_patch,black_patch])
        plte = 0
        pltr = 0
        plt.title('TP: ' + str("{0:.0f}".format(tp)) + ' TN: ' + str("{0:.0f}".format(tn)) + ' FP: ' + str("{0:.0f}".format(fp)) + ' FN: ' + str("{0:.0f}".format(fn)) )       
        for tst in testIndex:
            if(int(tst)==cnt):
                plte=1
        for tst in trainIndex:
            if(int(tst)==cnt):
                pltr=1
        if(plte==1):
            print 'TEEEEST ' + pullbacks[cnt]
            plt.xlabel(pullbacks[cnt] + ' TEST ' + basetitle)
            if(f1!=0):
                prec_all += prec
                rec_all += rec
                acc_all += acc
                cnt_all += 1
            else:
                sc+=1
                scts+=1
        elif(pltr==1):
            plt.xlabel(pullbacks[cnt] + ' TRAINING ' + basetitle)
            if(f1!=0):
                prec_train += prec
                rec_train += rec
                acc_train += acc
                cnt_train += 1
            else:
                sc+=1
                sctr+=1
        else:
            plt.xlabel(pullbacks[cnt] + ' RNN TRAINING ' + basetitle)
            print('RNNNNNNNNNNNNNNNNNNNNNNNNNNNNN')
            if(f1!=0):
                prec_extrain += prec
                rec_extrain += rec
                acc_extrain += acc
                cnt_extrain += 1
            else:
                sc+=1
                scex+=1
    	plt.ylabel('F1: ' + str("{0:.2f}".format(f1)) + ' Prec: ' + str("{0:.2f}".format(prec)) + ' Rec: ' + str("{0:.2f}".format(rec))+ ' Acc: ' + str("{0:.2f}".format(acc)))
    	axes.set_ylim([-0.5,1.5])
        plt.savefig(pullbacks[cnt]+'.png')
        cnt+=1
        plt.close()
    prec_all /= cnt_all
    rec_all /= cnt_all
    acc_all /= cnt_all
    prec_train /= cnt_train
    rec_train /= cnt_train
    acc_train /= cnt_train
    prec_extrain /= cnt_extrain
    rec_extrain /= cnt_extrain
    acc_extrain /= cnt_extrain
    print('_TEST_')
    print('Acc: ' + str(acc_all))
    print ('Prec: '+str(prec_all))
    print ('Rec: ' + str(rec_all))
    print('_ExTrain_')
    print('Acc: ' + str(acc_extrain))
    print ('Prec: ' + str(prec_extrain))
    print ('Rec: ' + str(rec_extrain))
    print('_TRAIN_')
    print('Acc: ' + str(acc_train))
    print('Prec: ' + str(prec_train))
    print('Rec: ' + str(rec_train))
    print('Ignored ' + str(sc) + ' special cases')
    print(str(sctr) + ' in training.')
    print(str(scts) + ' in test.')
    print(str(scex) + ' in Extra-Training')

def predicter():
    netfile = 'saved_models/0_4-BN-32-01UBCNN_Calcio_trained_model.h5'
    #netfile = 'saved_models/6_4-BN-64-04UBCNN_Calcio_trained_model.h5'
    #netfile = 'saved_models/1_4-BN-64-01UBCNN_Calcio_trained_model.h5'
    model = load_model(netfile, custom_objects={'f1': f1,'precision': precision,'recall': recall})
    print 'loaded ' + netfile
    last = ''
    count = 0
    output = []
    output2 = []
    for image in filenames:
        tmpi = image.split('/')[1]
        if(tmpi!=last):
            print tmpi
            if(count!=0):
                print 'appending ' + last + '...'
                outputAll.append(output)
            output = []
            last = tmpi
        image_shape = (128,128,1)
        batch_x = np.zeros((1,) + image_shape, dtype=K.floatx())
        img = load_img(image,grayscale=True)
        x = img_to_array(img)
        x/=255
        new_im = np.reshape(x,(512,512))
        new_im_small = resize(new_im, (128,128,1), order=1, preserve_range=True)
        batch_x[0] = new_im_small

        res = model.predict_classes(batch_x)
        res2 = model.predict(batch_x)
        #print image + ' was ' + str(res[0][0])
        output.append(res[0][0])
        output2.append(res2[0][0])
        count+=1
fileReader()
predicter()
plotter()

