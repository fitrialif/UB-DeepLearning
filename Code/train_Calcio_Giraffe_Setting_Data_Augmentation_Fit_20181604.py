from UBCNN import *
from UBMetrics import *
import keras
from keras import backend as k
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from keras.callbacks import ModelCheckpoint, Callback
from sklearn.model_selection import KFold
import sys
import csv

#
#It is needed the to run first the prepare_pullbacks_giraffe_setting.py
#

#Custom Metrics for UB-CNN-Calcium

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

np.random.seed(42)
print("THIIIIIIIIIIIIIIIIIIIIS")
reader = ImageReader()

name_pullbacks_x = 'PULLBACKS_X_GIRAFFE.csv'
name_pullbacks_y = 'PULLBACKS_Y_GIRAFFE.csv'
name_pullbacks_names = 'PULLBACKS_NAMES_GIRAFFE.csv'
allress = []
X,Y,start_indices,end_indices = reader.read_pullbacks_from_CSV(names=name_pullbacks_names,file_x=name_pullbacks_x,file_y=name_pullbacks_y)

X = X/255.0
Y = Y[:,2]#CALCIO
ecount=0
while (ecount<16):
    N_pullbacks = len(start_indices)

    print(N_pullbacks)

    indices_pullbacks = range(N_pullbacks)

    N_folds = 10
    kf = KFold(n_splits=N_folds, shuffle=True, random_state=42)

    tr_epochs = 100

    current_fold = 0

    basename ='KFOLDNETS_GIRAFFE_SETTING_CALCIO_RSNET1_'

    ts_accs = np.zeros((10,tr_epochs))
    tr_accs = np.zeros((10,tr_epochs))
    precisions = np.zeros((10,tr_epochs))
    recalls = np.zeros((10,tr_epochs))

    batch_size = 16
    num_classes = 1
    epochs = 100
    data_augmentation = True
    num_predictions = 20
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'UBCNN_Calcio_trained_model.h5'

    counter = 0

    for train_pullbacks, test_pullbacks in kf.split(indices_pullbacks):

    	frame_tr_indices_cnn = None
    	frame_tr_indices_rnn = None
    	frame_test_indices = None

    	nseq_train = len(train_pullbacks)
    	nseq_train_cnn = int(np.floor(nseq_train/2.0))
    	nseq_train_rnn = nseq_train-nseq_train_cnn

    	train_pullbacks_cnn = train_pullbacks[:nseq_train_cnn]
    	train_pullbacks_rnn = train_pullbacks[nseq_train_cnn:]

    	for idx in train_pullbacks_cnn:
    		new_indices = [int(i) for i in range(start_indices[idx],end_indices[idx]+1)]
    		if frame_tr_indices_cnn is None:
    			frame_tr_indices_cnn = new_indices
    		else:
    			frame_tr_indices_cnn = np.r_[frame_tr_indices_cnn,new_indices]

    	for idx in train_pullbacks_rnn:
    		new_indices = [int(i) for i in range(start_indices[idx],end_indices[idx]+1)]
    		if frame_tr_indices_rnn is None:
    			frame_tr_indices_rnn = new_indices
    		else:
    			frame_tr_indices_rnn = np.r_[frame_tr_indices_rnn,new_indices]

    	for idx in test_pullbacks:
    		new_indices = [int(i) for i in range(start_indices[idx],end_indices[idx]+1)]
    		if frame_test_indices is None:
    			frame_test_indices = new_indices
    		else:
    			frame_test_indices = np.r_[frame_test_indices,new_indices]

    	X_train_cnn = X[frame_tr_indices_cnn,:]
    	y_train_cnn = Y[frame_tr_indices_cnn]

        #Y_Train Conversion
    	#y_train_cnn = keras.utils.to_categorical(y_train_cnn, num_classes)

    	X_test = X[frame_test_indices,:]
    	y_test = Y[frame_test_indices]

        #Y_Test Conversion
    	#y_test = keras.utils.to_categorical(y_test, num_classes)

    	print("SHAPES: ")
    	print(X_train_cnn.shape)
    	print(y_train_cnn.shape)
    	print(X_test.shape)
    	print(y_test.shape)
    	print("LABELS")
    	print(y_train_cnn)

        #Model Definition
    	model = Sequential()
	if(ecount==0):
		model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 1), activation='relu'))
		model.add(BatchNormalization())
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.4))

		model.add(Conv2D(64, (3, 3), activation='relu'))
		model.add(BatchNormalization())
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.4))

		model.add(Conv2D(64, (3, 3), activation='relu'))
		model.add(BatchNormalization())
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.4))

		model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
		model.add(Dense(64, activation='relu'))
		model.add(Dropout(0.4))
		model.add(Dense(1, activation='sigmoid'))
		model.summary()
		model_name = '3-BN-32-04'+'UBCNN_Calcio_trained_model.h5'
		results_file = basename+'3-BN-32-04'+'%dEp_Fold%d_Of_%d.results'%(tr_epochs,current_fold,N_folds)
		filename = basename+'3-BN-32-04'+'%dEp_Fold%d_Of_%d.model'%(tr_epochs,current_fold,N_folds)
	elif(ecount==1):
	    	model.add(Conv2D(16, (3, 3), input_shape=(128, 128, 1), activation='relu'))
		model.add(BatchNormalization())
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.4))

		model.add(Conv2D(32, (3, 3), activation='relu'))
		model.add(BatchNormalization())
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.4))

		model.add(Conv2D(32, (3, 3), activation='relu'))
		model.add(BatchNormalization())
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.4))

		model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
		model.add(Dense(64, activation='relu'))
		model.add(Dropout(0.4))
		model.add(Dense(1, activation='sigmoid'))
		model.summary()
		results_file = basename+'3-BN-16-04'+'%dEp_Fold%d_Of_%d.results'%(tr_epochs,current_fold,N_folds)
		filename = basename+'3-BN-16-04'+'%dEp_Fold%d_Of_%d.model'%(tr_epochs,current_fold,N_folds)
		model_name = '3-BN-16-04'+'UBCNN_Calcio_trained_model.h5'
	elif(ecount==2):
	    	model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 1), activation='relu'))
		model.add(BatchNormalization())
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.1))

		model.add(Conv2D(64, (3, 3), activation='relu'))
		model.add(BatchNormalization())
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.1))

		model.add(Conv2D(64, (3, 3), activation='relu'))
		model.add(BatchNormalization())
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.1))

		model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
		model.add(Dense(64, activation='relu'))
		model.add(Dropout(0.1))
		model.add(Dense(1, activation='sigmoid'))
		model.summary()
		results_file = basename+'3-BN-32-01'+'%dEp_Fold%d_Of_%d.results'%(tr_epochs,current_fold,N_folds)
		filename = basename+'3-BN-32-01'+'%dEp_Fold%d_Of_%d.model'%(tr_epochs,current_fold,N_folds)
		model_name = '3-BN-32-01'+'UBCNN_Calcio_trained_model.h5'
	elif(ecount==3):
	    	model.add(Conv2D(16, (3, 3), input_shape=(128, 128, 1), activation='relu'))
		model.add(BatchNormalization())
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.1))

		model.add(Conv2D(32, (3, 3), activation='relu'))
		model.add(BatchNormalization())
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.1))

		model.add(Conv2D(32, (3, 3), activation='relu'))
		model.add(BatchNormalization())
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.1))

		model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
		model.add(Dense(64, activation='relu'))
		model.add(Dropout(0.1))
		model.add(Dense(1, activation='sigmoid'))
		model.summary()
		results_file = basename+'3-BN-16-01'+'%dEp_Fold%d_Of_%d.results'%(tr_epochs,current_fold,N_folds)
		filename = basename+'3-BN-16-01'+'%dEp_Fold%d_Of_%d.model'%(tr_epochs,current_fold,N_folds)
		model_name = '3-BN-16-01'+'UBCNN_Calcio_trained_model.h5'
	elif(ecount==4):
	    	model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 1), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.4))

		model.add(Conv2D(64, (3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.4))

		model.add(Conv2D(64, (3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.4))

		model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
		model.add(Dense(64, activation='relu'))
		model.add(Dropout(0.4))
		model.add(Dense(1, activation='sigmoid'))
		model.summary()
		results_file = basename+'3-32-04'+'%dEp_Fold%d_Of_%d.results'%(tr_epochs,current_fold,N_folds)
		filename = basename+'3-32-04'+'%dEp_Fold%d_Of_%d.model'%(tr_epochs,current_fold,N_folds)
		model_name = '3-32-04'+'UBCNN_Calcio_trained_model.h5'
	elif(ecount==5):
	    	model.add(Conv2D(16, (3, 3), input_shape=(128, 128, 1), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.4))

		model.add(Conv2D(32, (3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.4))

		model.add(Conv2D(32, (3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.4))

		model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
		model.add(Dense(64, activation='relu'))
		model.add(Dropout(0.4))
		model.add(Dense(1, activation='sigmoid'))
		model.summary()
		results_file = basename+'3-16-04'+'%dEp_Fold%d_Of_%d.results'%(tr_epochs,current_fold,N_folds)
		filename = basename+'3-16-04'+'%dEp_Fold%d_Of_%d.model'%(tr_epochs,current_fold,N_folds)
		model_name = '3-16-04'+'UBCNN_Calcio_trained_model.h5'
	elif(ecount==6):
	    	model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 1), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.1))

		model.add(Conv2D(64, (3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.1))

		model.add(Conv2D(64, (3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.1))

		model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
		model.add(Dense(64, activation='relu'))
		model.add(Dropout(0.4))
		model.add(Dense(1, activation='sigmoid'))
		model.summary()
		results_file = basename+'3-32-01'+'%dEp_Fold%d_Of_%d.results'%(tr_epochs,current_fold,N_folds)
		filename = basename+'3-32-01'+'%dEp_Fold%d_Of_%d.model'%(tr_epochs,current_fold,N_folds)
		model_name = '3-32-01'+'UBCNN_Calcio_trained_model.h5'
	elif(ecount==7):
	    	model.add(Conv2D(16, (3, 3), input_shape=(128, 128, 1), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.1))

		model.add(Conv2D(32, (3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.1))

		model.add(Conv2D(32, (3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.1))

		model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
		model.add(Dense(64, activation='relu'))
		model.add(Dropout(0.4))
		model.add(Dense(1, activation='sigmoid'))
		model.summary()
		results_file = basename+'3-16-01'+'%dEp_Fold%d_Of_%d.results'%(tr_epochs,current_fold,N_folds)
		filename = basename+'3-16-01'+'%dEp_Fold%d_Of_%d.model'%(tr_epochs,current_fold,N_folds)
		model_name = '3-16-01'+'UBCNN_Calcio_trained_model.h5'
	elif(ecount==8):
	    	model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 1), activation='relu'))
		model.add(BatchNormalization())
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.4))

	    	model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 1), activation='relu'))
		model.add(BatchNormalization())
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.4))

		model.add(Conv2D(64, (3, 3), activation='relu'))
		model.add(BatchNormalization())
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.4))

		model.add(Conv2D(64, (3, 3), activation='relu'))
		model.add(BatchNormalization())
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.4))

		model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
		model.add(Dense(64, activation='relu'))
		model.add(Dropout(0.4))
		model.add(Dense(1, activation='sigmoid'))
		model.summary()
		results_file = basename+'4-BN-32-04'+'%dEp_Fold%d_Of_%d.results'%(tr_epochs,current_fold,N_folds)
		filename = basename+'4-BN-32-04'+'%dEp_Fold%d_Of_%d.model'%(tr_epochs,current_fold,N_folds)
		model_name = '4-BN-32-04'+'UBCNN_Calcio_trained_model.h5'
	elif(ecount==9):
	    	model.add(Conv2D(16, (3, 3), input_shape=(128, 128, 1), activation='relu'))
		model.add(BatchNormalization())
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.4))

	    	model.add(Conv2D(16, (3, 3), input_shape=(128, 128, 1), activation='relu'))
		model.add(BatchNormalization())
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.4))

		model.add(Conv2D(32, (3, 3), activation='relu'))
		model.add(BatchNormalization())
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.4))

		model.add(Conv2D(32, (3, 3), activation='relu'))
		model.add(BatchNormalization())
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.4))

		model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
		model.add(Dense(64, activation='relu'))
		model.add(Dropout(0.4))
		model.add(Dense(1, activation='sigmoid'))
		model.summary()
		results_file = basename+'4-BN-16-04'+'%dEp_Fold%d_Of_%d.results'%(tr_epochs,current_fold,N_folds)
		filename = basename+'4-BN-16-04'+'%dEp_Fold%d_Of_%d.model'%(tr_epochs,current_fold,N_folds)
		model_name = '4-BN-16-04'+'UBCNN_Calcio_trained_model.h5'
	elif(ecount==10):
	    	model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 1), activation='relu'))
		model.add(BatchNormalization())
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.1))

	    	model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 1), activation='relu'))
		model.add(BatchNormalization())
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.1))

		model.add(Conv2D(64, (3, 3), activation='relu'))
		model.add(BatchNormalization())
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.1))

		model.add(Conv2D(64, (3, 3), activation='relu'))
		model.add(BatchNormalization())
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.1))

		model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
		model.add(Dense(64, activation='relu'))
		model.add(Dropout(0.1))
		model.add(Dense(1, activation='sigmoid'))
		model.summary()
		results_file = basename+'4-BN-32-01'+'%dEp_Fold%d_Of_%d.results'%(tr_epochs,current_fold,N_folds)
		filename = basename+'4-BN-32-01'+'%dEp_Fold%d_Of_%d.model'%(tr_epochs,current_fold,N_folds)
		model_name = '4-BN-32-01'+'UBCNN_Calcio_trained_model.h5'
	elif(ecount==11):
	    	model.add(Conv2D(16, (3, 3), input_shape=(128, 128, 1), activation='relu'))
		model.add(BatchNormalization())
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.1))

	    	model.add(Conv2D(16, (3, 3), input_shape=(128, 128, 1), activation='relu'))
		model.add(BatchNormalization())
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.1))

		model.add(Conv2D(32, (3, 3), activation='relu'))
		model.add(BatchNormalization())
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.1))

		model.add(Conv2D(32, (3, 3), activation='relu'))
		model.add(BatchNormalization())
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.1))

		model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
		model.add(Dense(64, activation='relu'))
		model.add(Dropout(0.1))
		model.add(Dense(1, activation='sigmoid'))
		model.summary()
		results_file = basename+'4-BN-16-01'+'%dEp_Fold%d_Of_%d.results'%(tr_epochs,current_fold,N_folds)
		filename = basename+'4-BN-16-01'+'%dEp_Fold%d_Of_%d.model'%(tr_epochs,current_fold,N_folds)
		model_name = '4-BN-16-01'+'UBCNN_Calcio_trained_model.h5'
	elif(ecount==12):
	    	model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 1), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.4))

	    	model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 1), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.4))

		model.add(Conv2D(64, (3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.4))

		model.add(Conv2D(64, (3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.4))

		model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
		model.add(Dense(64, activation='relu'))
		model.add(Dropout(0.4))
		model.add(Dense(1, activation='sigmoid'))
		model.summary()
		results_file = basename+'4-32-04'+'%dEp_Fold%d_Of_%d.results'%(tr_epochs,current_fold,N_folds)
		filename = basename+'4-32-04'+'%dEp_Fold%d_Of_%d.model'%(tr_epochs,current_fold,N_folds)
		model_name = '4-32-04'+'UBCNN_Calcio_trained_model.h5'
	elif(ecount==13):
	    	model.add(Conv2D(16, (3, 3), input_shape=(128, 128, 1), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.4))

	   	model.add(Conv2D(16, (3, 3), input_shape=(128, 128, 1), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.4))

		model.add(Conv2D(32, (3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.4))

		model.add(Conv2D(32, (3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.4))

		model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
		model.add(Dense(64, activation='relu'))
		model.add(Dropout(0.4))
		model.add(Dense(1, activation='sigmoid'))
		model.summary()
		results_file = basename+'4-16-04'+'%dEp_Fold%d_Of_%d.results'%(tr_epochs,current_fold,N_folds)
		filename = basename+'4-16-04'+'%dEp_Fold%d_Of_%d.model'%(tr_epochs,current_fold,N_folds)
		model_name = '4-16-04'+'UBCNN_Calcio_trained_model.h5'
	elif(ecount==14):
	    	model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 1), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.1))

	    	model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 1), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.1))

		model.add(Conv2D(64, (3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.1))

		model.add(Conv2D(64, (3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.1))

		model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
		model.add(Dense(64, activation='relu'))
		model.add(Dropout(0.1))
		model.add(Dense(1, activation='sigmoid'))
		model.summary()
		results_file = basename+'4-32-01'+'%dEp_Fold%d_Of_%d.results'%(tr_epochs,current_fold,N_folds)
		filename = basename+'4-32-01'+'%dEp_Fold%d_Of_%d.model'%(tr_epochs,current_fold,N_folds)
		model_name = '4-32-01'+'UBCNN_Calcio_trained_model.h5'
	elif(ecount==15):
	    	model.add(Conv2D(16, (3, 3), input_shape=(128, 128, 1), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.1))

	    	model.add(Conv2D(16, (3, 3), input_shape=(128, 128, 1), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.1))

		model.add(Conv2D(32, (3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.1))

		model.add(Conv2D(32, (3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.1))

		model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
		model.add(Dense(64, activation='relu'))
		model.add(Dropout(0.1))
		model.add(Dense(1, activation='sigmoid'))
		model.summary()
		results_file = basename+'4-16-01'+'%dEp_Fold%d_Of_%d.results'%(tr_epochs,current_fold,N_folds)
		filename = basename+'4-16-01'+'%dEp_Fold%d_Of_%d.model'%(tr_epochs,current_fold,N_folds)
		model_name = '4-16-01'+'UBCNN_Calcio_trained_model.h5'

    	# initiate RMSprop optimizer
    	opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    	model.compile(loss='binary_crossentropy',
    	              optimizer='Adam',
    	              metrics=[precision, recall, f1, 'accuracy'])

    	# smetrics = SingleLabelMonitor()

    	# net_name = 'SingleNet_Calcio_30CLEAN_UNION_FOLD%d'%current_fold
    	# target_vars = ['calcio']

    	# filename = basename+'%dEp_Fold%d_Of_%d.model'%(tr_epochs,current_fold,N_folds)
    	print('Using real-time data augmentation.')

        #Start Data Augmentation
    	datagen = ImageDataGenerator(
    	    featurewise_center=False,  # set input mean to 0 over the dataset
    	    samplewise_center=False,  # set each sample mean to 0
    	    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    	    samplewise_std_normalization=False,  # divide each input by its std
    	    zca_whitening=False,  # apply ZCA whitening
    	    rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
    	    width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
    	    height_shift_range=0,  # randomly shift images vertically (fraction of total height)
    	    horizontal_flip=True,  # randomly flip images
    	    vertical_flip=True)  # randomly flip images

    	datagen.fit(X_train_cnn)
        #Start Data Fitting With Augmented DataSet
    	history = model.fit_generator(datagen.flow(X_train_cnn, y_train_cnn,
                                         batch_size=16),
                            epochs=epochs,
                            validation_data=(X_test, y_test),
                            workers=4)



    	current_fold+=1
    	if not os.path.isdir(save_dir):
    	    os.makedirs(save_dir)
    	model_path = os.path.join(save_dir, model_name)
    	model.save(model_path)
    	print('Saved trained model at %s ' % model_path)

    	# Score trained model.
    	print("!!__________Score_________!!")
    	scores = model.evaluate(X_test, y_test, verbose=1)
    	print('Test loss:', scores[0])
    	print('Test accuracy:', scores[1])
    	del model

    # fmet = []
    # fmet.append(np.mean(tr_accs,axis=0))
    # print fmet[0]
    # fmet.append(np.mean(ts_accs,axis=0))
    # print fmet[1]
    # fmet.append(np.mean(precisions,axis=0))
    # print fmet[2]
    # fmet.append(np.mean(recalls,axis=0))
    # print fmet[3]
    # allress.append(fmet)
    # with open("results4Layer_20180321-01.csv","w+") as my_csv:
    #     csvWriter = csv.writer(my_csv,delimiter=',')
    #     csvWriter.writerows(allress)
