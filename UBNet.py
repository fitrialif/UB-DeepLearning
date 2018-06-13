from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.layers import Masking
from keras.models import load_model
from keras.optimizers import Adam
import csv
import numpy as np 
import pylab as plt
import pickle


class UBNet:

    def __init__(self):
        print("creating UBNet ... ")
        print("... UBNet created!")

    def compile_model(self,max_sequence_len,n_features,batch_size,NUnits_RNN=10,NUnits_Dense=0):

        self.batch_size = batch_size
        self.max_sequence_len = max_sequence_len
        self.n_features = n_features

        self.NUnits_RNN = NUnits_RNN
        self.NUnits_Dense = NUnits_Dense
        
        net = Sequential()
        net.add(Masking(mask_value=-1, batch_input_shape=(self.batch_size,self.max_sequence_len,self.n_features)))
        #By default concat mode is merge
        net.add(Bidirectional(LSTM(self.NUnits_RNN,return_sequences=True)))
        net.add(Bidirectional(LSTM(self.NUnits_RNN,return_sequences=True)))
        if NUnits_Dense > 0:
            net.add(TimeDistributed(Dense(self.NUnits_Dense, activation='relu')))
        net.add(TimeDistributed(Dense(n_features, activation='sigmoid')))

        print("compiling model ... ")

        optimizer = Adam()
        #optimizer = SGD(lr=0.01, momentum=0.5, decay=0.0, nesterov=True)
        net.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        print("... compiled! (details below)")
        
        self.model = net
        self.trained = False
        net.summary()

    def train_model(self,X,y,batch_size=16,n_epochs=1,Xt=[],yt=[],save_name=None):
        
        net = self.model

        print "training model ..."
        if len(Xt)!=0 and len(yt)!=0:
            net.fit(X,y,epochs=n_epochs,batch_size=batch_size,verbose=2,shuffle=False,validation_data= (Xt, yt))
        else:
            net.fit(X,y,epochs=n_epochs,batch_size=batch_size,verbose=2,shuffle=False)

        net.trained = True

        if save_name != None:
            net.save(save_name)

    def predict(self,x,batch_size):
        net = self.model
        y = net.predict(x,batch_size=batch_size)
        return y

class SequenceReader:
    
    def __init__(self):
        print("creating SeqReader ... ")
        print("... Reader created!")


    def check_length_seqs(self,filename,n_features,has_header=False):

        with open(filename, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            if has_header:
                next(reader)
            seq_lengths = dict()
            seq_indices = dict()
            for row in reader:
                name_elements = row[0]
                seq_name = str(name_elements)
                #frame_name = boh
                if seq_name in seq_lengths:
                    seq_lengths[seq_name]+=1
                else:
                    seq_lengths[seq_name]=1
                    seq_indices[seq_name] = len(seq_indices)

        lens = list(seq_lengths.values())
        max_len=max(lens)
        return max_len

    def read_data(self,filename,n_features,has_header=False,force_max_len=False,max_len_forced=0):

        with open(filename, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            if has_header:
                next(reader)
            seq_lengths = dict()
            seq_indices = dict()
            for row in reader:
                name_elements = row[0]
                seq_name = str(name_elements)
                #frame_name = boh
                if seq_name in seq_lengths:
                    seq_lengths[seq_name]+=1
                else:
                    seq_lengths[seq_name]=1
                    seq_indices[seq_name] = len(seq_indices)

        n_sequences = len(seq_lengths) 
 
        lens = list(seq_lengths.values())
        max_len=max(lens)

        if force_max_len:
            max_len = max_len_forced
            print("Forcing max lenght of sequenxces to %d"%max_len)

        data = -np.ones((n_sequences,max_len,2*n_features),dtype=np.float32) 
       
        with open(filename, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            if has_header:
                next(reader)
            pointers_write = np.zeros(n_sequences,dtype=np.int32)
            #print pointers_write
            for row in reader:
                name_elements = row[0]
                seq_name = str(name_elements)
                seq_length = seq_lengths[seq_name]
                seq_idx = seq_indices[seq_name]
                pos_write = pointers_write[seq_idx]
                data[seq_idx,pos_write,0:n_features] = np.array([float(i) for i in row[1:n_features+1]])
                data[seq_idx,pos_write,n_features:] = np.array([int(i) for i in row[n_features+1:2*n_features+1]])
                pointers_write[seq_idx] +=1
                #print [int(i) for i in row[1:]] 

        print pointers_write
        print "Number of Sequences = %d"%n_sequences
        print "SHAPE:",data.shape
        # print "... Max = %d"%max(lens)
        # print "... Average = %d"%np.mean(lens)
        # print "... Min = %d"%min(lens)

        #print data[0,0:-1,0]
        #print data[0,0:-1,1]
        #print data[0,0:-1,2]

        list_of_names = ["" for x in range(len(data))]
        for key in seq_indices:
            list_of_names[seq_indices[key]] = key

        return data,n_sequences,max_len,list_of_names


    def prepare_supervised(self,data,noise_level=0.3,copy_data=True,print_flag=False):

        n_seq, n_time_steps, n_features = data.shape
        noisy_data = np.array(data,copy=copy_data)

        for i in range(n_seq):
            for j in range(n_time_steps):
                #rand_array =  noise_level*np.random.randn(n_features)#ABSOLUTE GAUSSIAN NOISE
                rand_array =  np.random.uniform(0,noise_level,n_features)#UNIFORM NOISE
                for k in range(n_features):
                    if data[i,j,k] == 1:
                        noisy_data[i,j,k] = np.maximum(data[i,j,k] - np.abs(rand_array[k]),0)
                    elif data[i,j,k] == 0:
                        noisy_data[i,j,k] = np.minimum(data[i,j,k] + np.abs(rand_array[k]),1)

        #### Examine some cases

        if print_flag:
            for i in range(n_seq):

                f, ((ax1, ax2), (ax3, ax4), (ax5, ax6))  = plt.subplots(3, 2, sharey=True)

                ax1.plot(range(len(data[i,:,0])),data[i,:,0],color='r')
                ax3.plot(range(len(data[i,:,1])),data[i,:,1],color='g')
                ax5.plot(range(len(data[i,:,2])),data[i,:,2],color='b')

                ax2.plot(range(len(noisy_data[i,:,0])),noisy_data[i,:,0],color='r')
                ax4.plot(range(len(noisy_data[i,:,1])),noisy_data[i,:,1],color='g')
                ax6.plot(range(len(noisy_data[i,:,2])),noisy_data[i,:,2],color='b')
                
                if print_flag :
                    name = "Sequence%d.png"%i
                    plt.savefig(name,dpi=200)
                
                if i < 2:
                   plt.show()

                plt.close()

        return noisy_data

class ExperimentParams:

    def __init__(self,n_sequences,n_features,max_len,n_units_rnn,n_units_dense_extra,batch_size,nepochs):
        self.n_sequences = n_sequences
        self.n_features = n_features
        self.max_sequence_len = max_len
        self.n_units_rnn = n_units_rnn
        self.n_units_dense_extra = n_units_dense_extra
        self.batch_size = batch_size
        self.nepochs = nepochs

class ExperimentResults:

        def __init__(self,Ytrue,Yhat,loss=0,accuracy=0):
            self.Ytrue = Ytrue
            self.Yhat = Yhat
            self.loss = loss
            self.accuracy = accuracy

class LabelledSequences:

        def __init__(self,X,Y):
            
            self.X = X
            self.Y = Y
            n_sequences, n_time_steps, n_features_Y = Y.shape
            n_sequences, n_time_steps, n_features_X = X.shape
            self.n_sequences = n_sequences
            self.time_steps = n_time_steps
            self.n_features_input = n_features_X
            self.n_features_output = n_features_Y
