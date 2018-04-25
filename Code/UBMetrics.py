from keras.callbacks import ModelCheckpoint, Callback
import numpy as np

class MultiLabelMetrics:

	def accuracy_one_class(self,ytrue,ypred):
		_ypred = ypred
		_ypred[_ypred>=0.5] = 1.0
		_ypred[_ypred<0.5] = 0.0		
		_acc = 1.0-np.mean(np.abs(ytrue - _ypred))
		return _acc

	def average_accuracy(self, Ytrue,Ypred):
		n,K = Ytrue.shape		
		_accs = []
		for i in range(K):
			_accs.append(self.accuracy_one_class(Ytrue[:,i],Ypred[:,i]))
		return np.mean(np.asarray(_accs))


	def precision_recall_one_class(self,ytrue,ypred):
		_ypred = ypred
		_ypred[_ypred>=0.5] = 1.0
		_ypred[_ypred<0.5] = 0.0		
		pred_positives = np.sum(_ypred)
		true_positives = np.sum(ytrue)
		
		if pred_positives > 0:
			_prec = np.sum(np.multiply(ytrue,_ypred))/pred_positives
		else:
			_prec = 1.0

		if true_positives > 0:
			_recall = np.sum(np.multiply(ytrue,_ypred))/true_positives
		else:
			_recall = 1.0

		return _prec, _recall


	def average_precision_recall(self, Ytrue,Ypred):
		n,K = Ytrue.shape		
		precisions = []
		recalls = []
		for i in range(K):
			_prec,_rec =self.precision_recall_one_class(Ytrue[:,i],Ypred[:,i])
			precisions.append()
			recalls.append()

		return np.mean(np.asarray(precisions)),np.mean(np.asarray(recalls))

class MultiLabelMonitor(Callback):

	def on_train_begin(self, logs={}):

		self.acc_0 = []
		self.acc_1 = []
		self.acc_2 = []

		self.prec_0 = []
		self.prec_1 = []
		self.prec_2 = []

		self.rec_0 = []
		self.rec_1 = []
		self.rec_2 = []

		self.metrics = MultiLabelMetrics()

	def on_epoch_end(self, epoch, logs={}):

		predict = np.asarray(self.model.predict(self.validation_data[0]))
		targ = self.validation_data[1]

		_acc0 = self.metrics.accuracy_one_class(targ[:,0],predict[:,0])
		_acc1 = self.metrics.accuracy_one_class(targ[:,1],predict[:,1])
		_acc2 = self.metrics.accuracy_one_class(targ[:,2],predict[:,2])

		_prec0, _rec0 = self.metrics.precision_recall_one_class(targ[:,0],predict[:,0])
		_prec1, _rec1 = self.metrics.precision_recall_one_class(targ[:,1],predict[:,1])
		_prec2, _rec2 = self.metrics.precision_recall_one_class(targ[:,2],predict[:,2])

		self.acc_0.append(_acc0)
		self.acc_1.append(_acc1)
		self.acc_2.append(_acc2)

		self.prec_0.append(_prec0)
		self.prec_1.append(_prec1)
		self.prec_2.append(_prec2)

		self.rec_0.append(_rec0)
		self.rec_1.append(_rec1)
		self.rec_2.append(_rec2)


		print(" - acc0: %f - acc1: %f - acc2: %f "%(_acc0,_acc1,_acc2))
		print(" - prec0: %f - prec1: %f - prec2: %f "%(_prec0,_prec1,_prec2))
		print(" - rec0: %f - rec1: %f - rec2: %f "%(_rec0,_rec1,_rec2))

		return

class MultiLabelMonitorMIX(Callback):

	def on_train_begin(self, logs={}):

		self.acc_0 = []
		self.acc_1 = []
		self.acc_2 = []
		self.metrics = MultiLabelMetrics()

	def on_epoch_end(self, epoch, logs={}):

		predict = np.asarray(self.model.predict([self.validation_data[0],self.validation_data[0],self.validation_data[0]]))
		targ = self.validation_data[3]

		_acc0 = self.metrics.accuracy_one_class(targ[:,0],predict[:,0])
		_acc1 = self.metrics.accuracy_one_class(targ[:,1],predict[:,1])
		_acc2 = self.metrics.accuracy_one_class(targ[:,2],predict[:,2])

		self.acc_0.append(_acc0)
		self.acc_1.append(_acc1)
		self.acc_2.append(_acc2)

		print(" - acc0: %f - acc1: %f - acc2: %f "%(_acc0,_acc1,_acc2))

		return

class SingleLabelMonitor(Callback):

	def on_train_begin(self, logs={}):

		self.acc_val = []
		self.prec_val = []
		self.rec_val = []

		#self.acc_tr = []
		#self.prec_tr = []
		#self.rec_tr = []

		self.metrics = MultiLabelMetrics()

	def on_epoch_end(self, epoch, logs={}):

		predict = np.asarray(self.model.predict(self.validation_data[0]))
		targ = self.validation_data[1]

		#predict_tr = np.asarray(self.model.predict(self.X_train))
		#targ_tr = self.y_train

		_acc0 = self.metrics.accuracy_one_class(targ,predict)
		_prec0, _rec0 = self.metrics.precision_recall_one_class(targ,predict)

		#_acc0_tr = self.metrics.accuracy_one_class(targ_tr,predict_tr)
		#_prec0_tr, _rec0_tr = self.metrics.precision_recall_one_class(targ_tr,predict_tr)

		self.acc_val.append(_acc0)
		self.prec_val.append(_prec0)
		self.rec_val.append(_rec0)

		#self.acc_tr.append(_acc0_tr)
		#self.prec_tr.append(_prec0_tr)
		#self.rec_tr.append(_rec0_tr)

		print(" - val acc: %f - val prec: %f - val rec: %f "%(_acc0,_prec0,_rec0))
		#print " - tr acc: %f - tr prec: %f - tr rec: %f "%(_acc0_tr,_prec0_tr,_rec0_tr)

		return



