from sklearn.model_selection import KFold

from keras.models import load_model
from keras import losses

import csv
import numpy as np 
import sys
import os

import pickle

np.random.seed(42)

n_features = 3

n_units_rnn = 30
n_units_dense_extra = 20
nepochs = 256

base_folder = sys.argv[1]
base_folder_figs = base_folder+"LSTM_PREDICTIONS_DEEP_GIRAFFE"

exp_name = sys.argv[2]

Nfolds = 10

base_name_tr = 'KFOLDNETS_GIRAFFE_SETTING_CALCIO_RSNET1'
base_name_extra = 'KFOLDNETS_GIRAFFE_SETTING_CALCIO_RSNET1'
base_name_ts = 'KFOLDNETS_GIRAFFE_SETTING_CALCIO_RSNET1'
save_base = 'KFOLDNETS_GIRAFFE_SETTING_CALCIO_RSNET1'

for fold in range(Nfolds):

	results_file_train = save_base+'_'+exp_name+'_Fold%d_Of_%d.results'%(fold,10)
	results_file_test = save_base+'_'+exp_name+'_Fold%d_Of_%d.results'%(fold,10)

	fin_train = open(results_file_train, 'r')
	fin_test = open(results_file_test, 'r')

	results_train = pickle.load(fin_train)
	results_test = pickle.load(fin_test)

	print results_train

	Y_true_train = results_train.train_accuracy
	Y_true_test = results_test.train_accuracy

	Y_hat_train = results_train.train_accuracy
	Y_dec_train = np.array(Y_hat_train,copy=True)
	Y_dec_train[Y_dec_train>=0.5] = 1.0
	Y_dec_train[Y_dec_train<0.5] = 0.0

	Y_hat_test = results_test.train_accuracy
	Y_dec_test = np.array(Y_hat_test,copy=True)
	Y_dec_test[Y_dec_test>=0.5] = 1.0
	Y_dec_test[Y_dec_test<0.5] = 0.0

#	folder_figs_tr = base_folder_figs+"/FOLD%d/TRAIN/"%fold
#	folder_figs_ts = base_folder_figs+"/FOLD%d/TEST/"%fold

	folder_figs_tr = base_folder_figs+"/TRAIN/"
	folder_figs_ts = base_folder_figs+"/TEST/"

	Markers_Train = -np.ones_like(Xtrain)
	Markers_Test = -np.ones_like(Xtest)

	NoMarkers_Train = -np.ones_like(Xtrain)
	NoMarkers_Test = -np.ones_like(Xtest)

	Markers_Train_RNN = -np.ones_like(Xtrain)
	Markers_Test_RNN = -np.ones_like(Xtest)

	NoMarkers_Train_RNN = -np.ones_like(Xtrain)
	NoMarkers_Test_RNN = -np.ones_like(Xtest)


	for feat in range(len(Xtrain[0,0,:])):
		for s in range(len(Xtrain)):#for each sequence
			for p in range(len(Xtrain[s,:,feat])):
				if Ytrain[s,p,feat] >= 0:
					if (Xtrain[s,p,feat] >= 0.5) and (Ytrain[s,p,feat] >= 0.5):
						Markers_Train[s,p,feat] = 0.05
					if (Xtrain[s,p,feat] < 0.5) and (Ytrain[s,p,feat] < 0.5):
						Markers_Train[s,p,feat] = 0.05
					if (Xtrain[s,p,feat] >= 0.5) and (Ytrain[s,p,feat] < 0.5):
						NoMarkers_Train[s,p,feat] = 0.05
					if (Xtrain[s,p,feat] < 0.5) and (Ytrain[s,p,feat] >= 0.5):
						NoMarkers_Train[s,p,feat] = 0.05

					if (Y_hat_train[s,p,feat] >= 0.5) and (Ytrain[s,p,feat] >= 0.5):
						Markers_Train_RNN[s,p,feat] = 0.1
					if (Y_hat_train[s,p,feat] < 0.5) and (Ytrain[s,p,feat] < 0.5):
						Markers_Train_RNN[s,p,feat] = 0.1
					if (Y_hat_train[s,p,feat] >= 0.5) and (Ytrain[s,p,feat] < 0.5):
						NoMarkers_Train_RNN[s,p,feat] = 0.1
					if (Y_hat_train[s,p,feat] < 0.5) and (Ytrain[s,p,feat] >= 0.5):
						NoMarkers_Train_RNN[s,p,feat] = 0.1

	for feat in range(len(Xtest[0,0,:])):
		for s in range(len(Xtest)):#for each sequence
			for p in range(len(Xtest[s,:,feat])):
				if Ytest[s,p,feat] >= 0:
					if (Xtest[s,p,feat] >= 0.5) and (Ytest[s,p,feat] >= 0.5):
						Markers_Test[s,p,feat] = 0.05
					if (Xtest[s,p,feat] < 0.5) and (Ytest[s,p,feat] < 0.5):
						Markers_Test[s,p,feat] = 0.05
					if (Xtest[s,p,feat] >= 0.5) and (Ytest[s,p,feat] < 0.5):
						NoMarkers_Test[s,p,feat] = 0.05
					if (Xtest[s,p,feat] < 0.5) and (Ytest[s,p,feat] >= 0.5):
						NoMarkers_Test[s,p,feat] = 0.05

					if (Y_hat_test[s,p,feat] >= 0.5) and (Ytest[s,p,feat] >= 0.5):
						Markers_Test_RNN[s,p,feat] = 0.1
					if (Y_hat_test[s,p,feat] < 0.5) and (Ytest[s,p,feat] < 0.5):
						Markers_Test_RNN[s,p,feat] = 0.1
					if (Y_hat_test[s,p,feat] >= 0.5) and (Ytest[s,p,feat] < 0.5):
						NoMarkers_Test_RNN[s,p,feat] = 0.1
					if (Y_hat_test[s,p,feat] < 0.5) and (Ytest[s,p,feat] >= 0.5):
						NoMarkers_Test_RNN[s,p,feat] = 0.1

	if not os.path.exists(folder_figs_tr):
		os.makedirs(folder_figs_tr)

	if not os.path.exists(folder_figs_ts):
		os.makedirs(folder_figs_ts)

	for seq in range(len((Xtrain))):
	
		f = plt.figure(figsize=(8,8))
		name_seq = names_train[seq]+"-in-fold%d"%fold

		ax13 = f.add_axes([0.1, 0.65, 0.8, 0.22])
		ax12 = f.add_axes([0.1, 0.35, 0.8, 0.22])
		ax11 = f.add_axes([0.1, 0.05, 0.8, 0.22])

		ind_stent = next((x[0] for x in enumerate(Y_true_train[seq,:,0]) if x[1] < 0), -1)
		ind_fibro = next((x[0] for x in enumerate(Y_true_train[seq,:,1]) if x[1] < 0), -1)
		ind_calcio = next((x[0] for x in enumerate(Y_true_train[seq,:,2]) if x[1] < 0), -1)

		l0 = ax13.axhline(y=0.5,color='b',linestyle=':',lw=0.3)
		l31,= ax13.plot(range(len(Y_true_train[seq,:ind_stent,0])),Y_true_train[seq,:ind_stent,0],color='b',lw=1.4,label='Stent/True')
		l32,= ax13.plot(range(len(Y_hat_train[seq,:ind_stent,0])),Y_hat_train[seq,:ind_stent,0],linestyle='-',color='red',lw=1.2,label='Stent/RNN')		
		l33,= ax13.plot(range(len(Xtrain[seq,:ind_stent,0])),Xtrain[seq,:ind_stent,0],linestyle='--',color='b',lw=1,label='Stent/CNN')
		l34,= ax13.plot(range(len(Markers_Train[seq,:ind_stent,0])),Markers_Train[seq,:ind_stent,0],linestyle='None',marker='o',color='b',markersize=2)
		l35,= ax13.plot(range(len(NoMarkers_Train[seq,:ind_stent,0])),NoMarkers_Train[seq,:ind_stent,0],linestyle='None',marker='x',color='orange',markersize=4)
		l34,= ax13.plot(range(len(Markers_Train_RNN[seq,:ind_stent,0])),Markers_Train_RNN[seq,:ind_stent,0],linestyle='None',marker='o',color='b',markersize=2)
		l35,= ax13.plot(range(len(NoMarkers_Train_RNN[seq,:ind_stent,0])),NoMarkers_Train_RNN[seq,:ind_stent,0],linestyle='None',marker='x',color='red',markersize=4)


		l0 = ax12.axhline(y=0.5,color='g',linestyle=':',lw=0.3)
		l21,= ax12.plot(range(len(Y_true_train[seq,:ind_fibro,1])),Y_true_train[seq,:ind_fibro,1],color='g',lw=1.4,label='Fibro/True')
		l22,= ax12.plot(range(len(Y_hat_train[seq,:ind_fibro,1])),Y_hat_train[seq,:ind_fibro,1],linestyle='-',color='red',lw=1.2,label='Fibro/RNN')		
		l23,= ax12.plot(range(len(Xtrain[seq,:ind_fibro,1])),Xtrain[seq,:ind_fibro,1],linestyle='--',color='g',lw=1,label='Fibro/CNN')
		l24,= ax12.plot(range(len(Markers_Train[seq,:ind_fibro,1])),Markers_Train[seq,:ind_fibro,1],linestyle='None',marker='o',color='g',markersize=2)
		l25,= ax12.plot(range(len(NoMarkers_Train[seq,:ind_fibro,1])),NoMarkers_Train[seq,:ind_fibro,1],linestyle='None',marker='x',color='orange',markersize=4)
		l24,= ax12.plot(range(len(Markers_Train_RNN[seq,:ind_fibro,1])),Markers_Train_RNN[seq,:ind_fibro,1],linestyle='None',marker='o',color='g',markersize=2)
		l25,= ax12.plot(range(len(NoMarkers_Train_RNN[seq,:ind_fibro,1])),NoMarkers_Train_RNN[seq,:ind_fibro,1],linestyle='None',marker='x',color='r',markersize=4)

		l0 = ax11.axhline(y=0.5,color='k',linestyle=':',lw=0.3)
		l11,= ax11.plot(range(len(Y_true_train[seq,:ind_calcio,2])),Y_true_train[seq,:ind_calcio,2],color='k',lw=1.4,label='Calci/True')
		l12,= ax11.plot(range(len(Y_hat_train[seq,:ind_calcio,2])),Y_hat_train[seq,:ind_calcio,2],linestyle='-',color='red',lw=1.2,label='Calcio/RNN')		
		l13,= ax11.plot(range(len(Xtrain[seq,:ind_calcio,2])),Xtrain[seq,:ind_calcio,2],linestyle='--',color='k',lw=1,label='Calci/CNN')
		l14,= ax11.plot(range(len(Markers_Train[seq,:ind_calcio,2])),Markers_Train[seq,:ind_calcio,2],linestyle='None',marker='o',color='k',markersize=2)
		l15,= ax11.plot(range(len(NoMarkers_Train[seq,:ind_calcio,2])),NoMarkers_Train[seq,:ind_calcio,2],linestyle='None',marker='x',color='orange',markersize=4)
		l14,= ax11.plot(range(len(Markers_Train_RNN[seq,:ind_calcio,2])),Markers_Train_RNN[seq,:ind_calcio,2],linestyle='None',marker='o',color='k',markersize=2)
		l15,= ax11.plot(range(len(NoMarkers_Train_RNN[seq,:ind_calcio,2])),NoMarkers_Train_RNN[seq,:ind_calcio,2],linestyle='None',marker='x',color='r',markersize=4)

		ax13.set_ylim((0,1.1))
		ax12.set_ylim((0,1.1))
		ax11.set_ylim((0,1.1))
		
		f.legend(bbox_to_anchor=(0.1,0.92,0.8, .08), loc=3,ncol=3, mode="expand", borderaxespad=0.,fontsize=8)

		plt.savefig(folder_figs_tr+name_seq+".pdf",dpi=200)
	 
		plt.close()

	for seq in range(len((Xtest))):
		
		f = plt.figure(figsize=(8,8))
		name_seq = names_test[seq]

		ax13 = f.add_axes([0.1, 0.65, 0.8, 0.22])
		ax12 = f.add_axes([0.1, 0.35, 0.8, 0.22])
		ax11 = f.add_axes([0.1, 0.05, 0.8, 0.22])

		ind_stent = next((x[0] for x in enumerate(Y_true_test[seq,:,0]) if x[1] < 0), -1)
		ind_fibro = next((x[0] for x in enumerate(Y_true_test[seq,:,1]) if x[1] < 0), -1)
		ind_calcio = next((x[0] for x in enumerate(Y_true_test[seq,:,2]) if x[1] < 0), -1)

		l0 = ax13.axhline(y=0.5,color='b',linestyle=':',lw=0.3)
		l31,= ax13.plot(range(len(Y_true_test[seq,:ind_stent,0])),Y_true_test[seq,:ind_stent,0],color='b',lw=1.4,label='Stent/True')
		l32,= ax13.plot(range(len(Y_hat_test[seq,:ind_stent,0])),Y_hat_test[seq,:ind_stent,0],linestyle='-',color='r',lw=1.2,label='Stent/RNN')		
		l33,= ax13.plot(range(len(Xtest[seq,:ind_stent,0])),Xtest[seq,:ind_stent,0],linestyle='--',color='b',lw=1,label='Stent/CNN')
		l34,= ax13.plot(range(len(Markers_Test[seq,:ind_stent,0])),Markers_Test[seq,:ind_stent,0],linestyle='None',marker='o',color='b',markersize=2)
		l35,= ax13.plot(range(len(NoMarkers_Test[seq,:ind_stent,0])),NoMarkers_Test[seq,:ind_stent,0],linestyle='None',marker='x',color='orange',markersize=4)
		l34,= ax13.plot(range(len(Markers_Test_RNN[seq,:ind_stent,0])),Markers_Test_RNN[seq,:ind_stent,0],linestyle='None',marker='o',color='b',markersize=2)
		l35,= ax13.plot(range(len(NoMarkers_Test_RNN[seq,:ind_stent,0])),NoMarkers_Test_RNN[seq,:ind_stent,0],linestyle='None',marker='x',color='red',markersize=4)


		l0 = ax12.axhline(y=0.5,color='g',linestyle=':',lw=0.3)			
		l21,= ax12.plot(range(len(Y_true_test[seq,:ind_fibro,1])),Y_true_test[seq,:ind_fibro,1],color='g',lw=1.4,label='Fibro/True')
		l22,= ax12.plot(range(len(Y_hat_test[seq,:ind_fibro,1])),Y_hat_test[seq,:ind_fibro,1],linestyle='-',color='r',lw=1.2,label='Fibro/RNN')		
		l23,= ax12.plot(range(len(Xtest[seq,:ind_fibro,1])),Xtest[seq,:ind_fibro,1],linestyle='--',color='g',lw=1,label='Fibro/CNN')
		l24,= ax12.plot(range(len(Markers_Test[seq,:ind_fibro,1])),Markers_Test[seq,:ind_fibro,1],linestyle='None',marker='o',color='g',markersize=2)
		l25,= ax12.plot(range(len(NoMarkers_Test[seq,:ind_fibro,1])),NoMarkers_Test[seq,:ind_fibro,1],linestyle='None',marker='x',color='orange',markersize=4)
		l24,= ax12.plot(range(len(Markers_Test_RNN[seq,:ind_fibro,1])),Markers_Test_RNN[seq,:ind_fibro,1],linestyle='None',marker='o',color='g',markersize=2)
		l25,= ax12.plot(range(len(NoMarkers_Test_RNN[seq,:ind_fibro,1])),NoMarkers_Test_RNN[seq,:ind_fibro,1],linestyle='None',marker='x',color='r',markersize=4)

		l0 = ax11.axhline(y=0.5,color='k',linestyle=':',lw=0.3)
		l11,= ax11.plot(range(len(Y_true_test[seq,:ind_calcio,2])),Y_true_test[seq,:ind_calcio,2],color='k',lw=1.4,label='Calci/True')
		l12,= ax11.plot(range(len(Y_hat_test[seq,:ind_calcio,2])),Y_hat_test[seq,:ind_calcio,2],linestyle='-',color='r',lw=1.2,label='Calcio/RNN')		
		l13,= ax11.plot(range(len(Xtest[seq,:ind_calcio,2])),Xtest[seq,:ind_calcio,2],linestyle='--',color='k',lw=1,label='Calci/CNN')
		l14,= ax11.plot(range(len(Markers_Test[seq,:ind_calcio,2])),Markers_Test[seq,:ind_calcio,2],linestyle='None',marker='o',color='k',markersize=2)
		l15,= ax11.plot(range(len(NoMarkers_Test[seq,:ind_calcio,2])),NoMarkers_Test[seq,:ind_calcio,2],linestyle='None',marker='x',color='orange',markersize=4)
		l14,= ax11.plot(range(len(Markers_Test_RNN[seq,:ind_calcio,2])),Markers_Test_RNN[seq,:ind_calcio,2],linestyle='None',marker='o',color='k',markersize=2)
		l15,= ax11.plot(range(len(NoMarkers_Test_RNN[seq,:ind_calcio,2])),NoMarkers_Test_RNN[seq,:ind_calcio,2],linestyle='None',marker='x',color='r',markersize=4)


		ax13.set_ylim((0,1.1))
		ax12.set_ylim((0,1.1))
		ax11.set_ylim((0,1.1))
	
		f.legend(bbox_to_anchor=(0.1,0.92,0.8, .08), loc=3,ncol=3, mode="expand", borderaxespad=0.,fontsize=8)
  
		plt.savefig(folder_figs_ts+name_seq+".pdf",dpi=200)
	 
		plt.close()



