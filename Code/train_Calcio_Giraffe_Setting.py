from UBCNN import *
from UBMetrics import *
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, Callback
from sklearn.model_selection import KFold
import sys
import csv

np.random.seed(42)

reader = ImageReader()

name_pullbacks_x = 'PULLBACKS_X_GIRAFFE.csv'
name_pullbacks_y = 'PULLBACKS_Y_GIRAFFE.csv'
name_pullbacks_names = 'PULLBACKS_NAMES_GIRAFFE.csv'
allress = []
for i in range(2, 4):
	fisize = pow(2,i)
	for j in range (0, 7):
		nfilter = pow(2,j)
		print("Filter Size: ")
		print(fisize)
		print("Number of Filters: ")
		print(nfilter)
		X,Y,start_indices,end_indices = reader.read_pullbacks_from_CSV(names=name_pullbacks_names,file_x=name_pullbacks_x,file_y=name_pullbacks_y)

		X = X/255.0
		Y = Y[:,2]#CALCIO

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
			X_test = X[frame_test_indices,:]
			y_test = Y[frame_test_indices]

			print("SHAPES: ")
			print(X_train_cnn.shape)
			print(y_train_cnn.shape)
			print(X_test.shape)
			print(y_test.shape)

			net = ThomasNet_Calcio()

			smetrics = SingleLabelMonitor()

			net.compile_model(input_shape=(128,128),n_target_feat=1,nfilters=nfilter,fsize=fisize)

			net_name = 'SingleNet_Calcio_30CLEAN_UNION_FOLD%d'%current_fold
			target_vars = ['calcio']

			filename = basename+'%dEp_Fold%d_Of_%d_nf%d_fs%d.model'%(tr_epochs,current_fold,N_folds,nfilter,fisize)

			history = net.train_model(X_train_cnn, y_train_cnn, X_val=X_test, y_val=y_test, batch_size=16, n_epochs=tr_epochs,save_name=filename,callbacks=[smetrics])

			history_tr_acc = history.history['acc']
			history_val_acc = history.history['val_acc']


			shape_X_train = X_train_cnn.shape
			shape_Y_train = y_train_cnn.shape

			params = ExperimentParams(net_name,shape_X_train,shape_Y_train,target_vars,tr_epochs)
			param_file = basename+'%dEp_Fold%d_Of_%d_nf%d_fs%d'%(tr_epochs,current_fold,N_folds,nfilter,fisize)

			with open(param_file, 'wb') as output:
				pickle.dump(params, output)

			test_error = net.test_model(X_test,y_test)
			train_error = net.test_model(X_train_cnn,y_train_cnn)

			print test_error
			print train_error

			others = dict()

			others['acc_val']= smetrics.acc_val
			#others['acc_tr']= smetrics.acc_tr

			others['prec_val']= smetrics.prec_val
			#others['prec_tr']= smetrics.prec_tr

			others['rec_val']= smetrics.rec_val
			#others['rec_tr']= smetrics.rec_tr

			tr_accs[counter,:] = history_tr_acc
			ts_accs[counter,:] = smetrics.acc_val
			precisions[counter,:] = np.array(smetrics.prec_val)
			recalls[counter,:] = np.array(smetrics.rec_val)

			counter +=1

			results = ExperimentResults(train_error,test_error,history_tr_acc,history_val_acc,others)

			results_file = basename+'%dEp_Fold%d_Of_%d.results'%(tr_epochs,current_fold,N_folds)

			with open(results_file, 'wb') as output:
				pickle.dump(results, output)

			current_fold+=1
			del net

		fmet = []
		fmet.append(nfilter)
		fmet.append(fisize)
		fmet.append(np.mean(tr_accs,axis=0))
		print fmet[2]
		fmet.append(np.mean(ts_accs,axis=0))
		print fmet[3]
		fmet.append(np.mean(precisions,axis=0))
		print fmet[4]
		fmet.append(np.mean(recalls,axis=0))
		print fmet[5]
		allress.append(fmet)


with open("resultsAll.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(allress)
