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

#Custom metrics redefinition in order to load the trained models
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

cntarg = 2
cntap = 0 
basetitle = []
bestfold = []
nets = int(sys.argv[1])

#Validate the model and model's metadata is available in the directory 
print('_Start validation')
while(cntap<nets):
    basetitle.append(str(sys.argv[cntarg]))
    cntarg+=1
    bestfold.append(int(sys.argv[cntarg]))
    cntarg+=1
    cntap+=1
print('_End validation')

for ip in range(nets):
    basetxt = 'trained_networks/'
    basetxt = basetxt + basetitle[ip] + '_UB_Calcium_PreRec.csv'
    with open(basetxt) as filename:
            template=csv.reader(filename)
            for row in template:
                print(row)
    print(basetitle)
    print(bestfold)

#Variables used to save the global scores of the evaluated networks
test_acc = []
train_acc = []
etrain_acc =[]
test_prec = []
train_prec = []
etrain_prec = []
test_rec = []
train_rec = []
etrain_rec = []
test_f1 = []
train_f1 = []
etrain_f1 = []
pullbacks = []
filenames = []
labelsP = []

#Labels Reader
with open("trained_networks/clean_labels_giraffe.csv") as filename:
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
with open("trained_networks/clean_labels_giraffe.csv") as filename:
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

#Loop for all evaluated networks
for ixps in range(nets):
    basetxt = 'trained_networks/'
    basetxt = basetxt + basetitle[ixps] + '_UB_Calcium_PreRec.csv'
    theAcc = []
    thePrec = []
    theRec = []
    theTTE = []
    #Keras Metrics Reader
    with open(basetxt) as filename:
        template=csv.reader(filename)
        for row in template:
            rowData = []
            columnC=0
            for column in row:
                if(columnC==0):
                    theAcc.append(float(column))
                if(columnC==1):
                    thePrec.append(float(column))
                if(columnC==2):
                    theRec.append(float(column))  
                columnC+=1
    all_test_acc = []
    all_train_acc = []
    all_etrain_acc =[]
    all_test_prec = []
    all_train_prec = []
    all_etrain_prec = []
    all_test_rec = []
    all_train_rec = []
    all_etrain_rec = []
    all_test_f1 = []
    all_train_f1 = []
    all_etrain_f1 = []
    #Loop for each network epoch
    for i in range(10):  
        images = []
        preFilen = []
        filenFil = []
        labelFil = []
        outputAll = []
        outputAll2 = []
        testIndex = []
        trainIndex = []
        istest = []
        testPath = []
        trainPath = []
        accAll = []
        precAll = []
        recAll = []
        print('_____________________Fold: ' + str(i) + '_____________________')
        print('________________' + basetitle[ixps] + '________________')
        netfile = 'trained_networks/saved_models/' + str(i) + '_' + basetitle[ixps] + 'UBCNN_Calcio_trained_model.h5'
        testFile = 'trained_networks/saved_models/' + str(i) + '_TEST_' + basetitle[ixps] + 'UBCNN_Calcio_trained_model.h5.csv'
        trainFile = 'trained_networks/saved_models/' + str(i) + '_TRAIN_' + basetitle[ixps] + 'UBCNN_Calcio_trained_model.h5.csv'
        # ----------------------------fileReader(testFile, trainFile) ---------------------------------------------

        #Test Pullback index reader
        with open(testFile) as filename:
            template=csv.reader(filename)
            for row in template:
                rowData = []
                columnC=0
                for column in row:
                    tmp =int(column)
                    testIndex.append(tmp)
        for tsti in testIndex:
            testPath.append(pullbacks[int(tsti)])

        #Train Pullback index reader
        with open(trainFile) as filename:
            template=csv.reader(filename)
            for row in template:
                rowData = []
                columnC=0
                for column in row:
                    tmp =int(column)
                    trainIndex.append(tmp)
        for tsti in trainIndex:
            trainPath.append(pullbacks[int(tsti)])

        #Read image, sends it to the pretrained network and gets the prediction

        model = load_model(netfile, custom_objects={'f1': f1,'precision': precision,'recall': recall})
        print 'loaded ' + netfile
        last = ''
        count = 0
        output = []
        output2 = []
        print len(labelsP)
        for image in filenames:
            tmpi = image.split('/')[1]
            if(tmpi!=last):
                print tmpi
                if(count!=0):
                    print 'appending ' + last + '...'
                    outputAll.append(output)
                    outputAll2.append(output2)
                output = []
                output2 = []
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
            output.append(res[0][0])
            output2.append(res2[0][0])
            count+=1

        # Ground Truth vs Prediction Plotter
        cnt_all = 0
        cnt_train = 0
        cnt_extrain = 0
        prec_all = []
        prec_train = []
        prec_extrain = []
        rec_all = []
        rec_train = []
        rec_extrain = []
        acc_all = []
        acc_train = []
        acc_extrain = []
        f1_all = []
        f1_train = []
        f1_extrain = []
        cnt = 0
        sc = 0
        sctr = 0
        scts = 0
        scex = 0
        for pullback in outputAll:
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
                theTTE.append(1)
                plt.xlabel(pullbacks[cnt] + ' TEST ' + basetitle[ixps])
                if(f1!=0):
                    prec_all.append(prec)
                    rec_all.append(rec)
                    acc_all.append(acc)
                    f1_all.append(f1)
                    cnt_all += 1
                else:
                    sc+=1
                    scts+=1
            elif(pltr==1):
                theTTE.append(0)
                plt.xlabel(pullbacks[cnt] + ' TRAINING ' + basetitle[ixps])
                if(f1!=0):
                    prec_train.append(prec)
                    rec_train.append(rec)
                    acc_train.append(acc)
                    f1_train.append(f1)
                else:
                    sc+=1
                    sctr+=1
            else:
                plt.xlabel(pullbacks[cnt] + ' RNN TRAINING ' + basetitle[ixps])
                theTTE.append(2)
                if(f1!=0):
                    prec_extrain.append(prec)
                    rec_extrain.append(rec)
                    acc_extrain.append(acc)
                    f1_extrain.append(f1)
                    cnt_extrain += 1
                else:
                    sc+=1
                    scex+=1
            plt.ylabel('F1: ' + str("{0:.2f}".format(f1)) + ' Prec: ' + str("{0:.2f}".format(prec)) + ' Rec: ' + str("{0:.2f}".format(rec))+ ' Acc: ' + str("{0:.2f}".format(acc)))
            axes.set_ylim([-0.5,1.5])
            if not os.path.exists(basetitle[ixps] + '/' + str(i) + '/Graficas/'):
                os.makedirs(basetitle[ixps] + '/' + str(i) + '/Graficas/')
            plt.savefig(basetitle[ixps] + '/' + str(i) + '/Graficas/' + pullbacks[cnt]+'.png')
            cnt+=1
            plt.close()

        #Append the metrics to the general array

        prec_all = np.array(prec_all)
        rec_all = np.array(rec_all)
        acc_all = np.array(acc_all)
        f1_all = np.array(f1_all)
        prec_train = np.array(prec_train)
        rec_train = np.array(rec_train)
        acc_train = np.array(acc_train)
        f1_train = np.array(f1_all)
        prec_extrain = np.array(prec_extrain)
        rec_extrain = np.array(rec_extrain)
        acc_extrain = np.array(acc_extrain)
        f1_extrain = np.array(f1_extrain)
        if not os.path.exists(basetitle[ixps] + '/' + str(i) + '/Resultados/'):
                os.makedirs(basetitle[ixps] + '/' + str(i) + '/Resultados/')
        if(i == bestfold[ixps]):
            text_file = open(basetitle[ixps] + '/' + str(i) + '/Resultados/'+ "BEST_Scores.txt", "w")
        else:
            text_file = open(basetitle[ixps] + '/' + str(i) + '/Resultados/'+ "Scores.txt", "w")
        text_file.write('_KERAS_' + '\n')
        text_file.write('Acc: ' + str(theAcc[i]) + '\n')
        text_file.write ('Prec: ' + str(thePrec[i]) + '\n')
        text_file.write ('Rec: ' + str(theRec[i]) + '\n')
        text_file.write('_TEST_' + '\n')
        text_file.write('Acc mean: ' + str(np.mean(acc_all)) + ' Acc std: ' + str(np.std(acc_all)) + '\n')
        all_test_acc.append(np.mean(acc_all))
        text_file.write ('Prec mean: '+str(np.mean(prec_all)) + ' Prec std: '+str(np.std(prec_all)) + '\n')
        all_test_prec.append(np.mean(prec_all))
        text_file.write ('Rec mean: ' + str(np.mean(rec_all)) + ' Rec std: ' + str(np.std(rec_all)) + '\n')
        all_test_rec.append(np.mean(rec_all))
        all_test_f1.append(np.mean(f1_all))
        text_file.write('_ExTrain_' + '\n')
        text_file.write('Acc mean: ' + str(np.mean(acc_extrain)) + 'Acc std: ' + str(np.std(acc_extrain)) + '\n')
        all_etrain_acc.append(np.mean(acc_extrain))
        text_file.write ('Prec mean: ' + str(np.mean(prec_extrain)) + ' Prec std: ' + str(np.std(prec_extrain)) + '\n')
        all_etrain_prec.append(np.mean(prec_extrain))
        text_file.write ('Rec mean: ' + str(np.mean(rec_extrain)) + ' Rec std: ' + str(np.std(rec_extrain)) + '\n')
        all_etrain_rec.append(np.mean(rec_extrain))
        all_etrain_f1.append(np.mean(f1_extrain))
        text_file.write('_TRAIN_' + '\n')
        text_file.write('Acc mean: ' + str(np.mean(acc_train)) + ' Acc std: ' + str(np.std(acc_train)) + '\n')
        all_train_acc.append(np.mean(acc_train))
        text_file.write('Prec mean: ' + str(np.mean(prec_train)) + ' Prec std: ' + str(np.std(prec_train)) + '\n')
        all_train_prec.append(np.mean(prec_train))
        text_file.write('Rec mean: ' + str(np.mean(rec_train)) + ' Rec std: ' + str(np.std(rec_train)) + '\n')
        all_train_rec.append(np.mean(rec_train))
        all_train_f1.append(np.mean(f1_train))
        text_file.write('Ignored ' + str(sc) + ' special cases' + '\n')
        text_file.write(str(scts) + ' in test.' + '\n')
        text_file.write(str(scex) + ' in Extra-Training' + '\n')
        text_file.close()

    # Save the Network Prediction in folders corrresponding if the pullbacvk was train test or extratrain

        cnt = 0
        stro = ''
        if not os.path.exists(basetitle[ixps] + '/' + str(i) + '/TRAIN/TRAIN/'):
            os.makedirs(basetitle[ixps] + '/' + str(i) + '/CSV/TRAIN/')
        if not os.path.exists(basetitle[ixps] + '/' + str(i) + '/CSV/TEST/'):
            os.makedirs(basetitle[ixps] + '/' + str(i) + '/CSV/TEST/')
        if not os.path.exists(basetitle[ixps] + '/' + str(i) + '/CSV/EXTRAIN/'):
            os.makedirs(basetitle[ixps] + '/' + str(i) + '/CSV/EXTRAIN/')
        for prediction in outputAll:
            if(theTTE[cnt]==0):            
                stro='TRAIN/'
            if(theTTE[cnt]==1):
                stro='TEST/'
            if(theTTE[cnt]==2):
                stro='EXTRAIN/'
            print ('Saved at ' + basetitle[ixps] + '/' + str(i) + '/CSV/' + stro + 'groud_truth_' + pullbacks[cnt]+'.csv')
            with open(basetitle[ixps] + '/' + str(i) + '/CSV/' + stro + 'groud_truth_' + pullbacks[cnt]+'.csv',"w+") as my_csv:
                csvWriter = csv.writer(my_csv,delimiter=',')
                cntEach = 0
                for frame in prediction:
                    csvWriter.writerow([outputAll2[cnt][cntEach],labelsP[cnt][cntEach]])
                    cntEach+=1
            cnt+=1
            print('Saved csv in ' + pullbacks[cnt]+'.csv')
    test_acc.append(all_test_acc)
    train_acc.append(all_train_acc)
    etrain_acc.append(all_etrain_acc)
    test_prec.append(all_test_prec)
    train_prec.append(all_train_prec)
    etrain_prec.append(all_etrain_prec)
    test_rec.append(all_test_rec)
    train_rec.append(all_train_rec)
    etrain_rec.append(all_etrain_rec)
    test_f1.append(all_test_f1)
    train_f1.append(all_train_f1)
    etrain_f1.append(all_etrain_f1)
# -------------------------------------boxplotter---------------------------------------------------

spread = np.random.rand(50) * 100
center = np.ones(25) * 50
flier_high = np.random.rand(10) * 100 + 100
flier_low = np.random.rand(10) * -100
data = np.concatenate((spread, center, flier_high, flier_low), 0)
labels = (basetitle)
# test_acc
plt.boxplot(test_acc)
plt.xticks(np.arange(len(labels))+1,labels)
plt.title('Test Acc')
plt.show()
    # test_acc
plt.boxplot(train_acc)
plt.xticks(np.arange(len(labels))+1,labels)
plt.title('Train Acc')
plt.show()
    # test_acc
plt.boxplot(etrain_acc)
plt.xticks(np.arange(len(labels))+1,labels)
plt.title('Extra-Training Acc')
plt.show()
    # test_acc
plt.boxplot(test_prec)
plt.xticks(np.arange(len(labels))+1,labels)
plt.title('Test Prec')
plt.show()
    # test_acc
plt.boxplot(train_prec)
plt.xticks(np.arange(len(labels))+1,labels)
plt.title('Train Prec')
plt.show()
    # test_acc
plt.boxplot(etrain_prec)
plt.xticks(np.arange(len(labels))+1,labels)
plt.title('Extra-Training Prec')
plt.show()
    # test_acc
plt.boxplot(test_rec)
plt.xticks(np.arange(len(labels))+1,labels)
plt.title('Test Rec')
plt.show()
    # test_acc
plt.boxplot(train_rec)
plt.xticks(np.arange(len(labels))+1,labels)
plt.title('Train Rec')
plt.show()
    # test_acc
plt.boxplot(etrain_rec)
plt.xticks(np.arange(len(labels))+1,labels)
plt.title('Extra-Training Rec')
plt.show()
    # test_acc
plt.boxplot(test_f1)
plt.xticks(np.arange(len(labels))+1,labels)
plt.title('Test F1')
plt.show()
    # test_acc
plt.boxplot(train_f1)
plt.xticks(np.arange(len(labels))+1,labels)
plt.title('Train F1')
plt.show()
    # test_acc
plt.boxplot(etrain_f1)
plt.xticks(np.arange(len(labels))+1,labels)
plt.title('Extra-Training F1')
plt.show()