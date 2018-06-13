from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Dense, Concatenate
from keras.models import load_model, Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization

from keras.preprocessing import image
from keras.models import load_model

import csv
import os
import sys
import numpy as np
import pickle

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array
from keras.preprocessing.image import load_img

from keras import backend as K

def create_weighted_binary_crossentropy(zero_weight, one_weight):

    def weighted_binary_crossentropy(y_true, y_pred):

        # Original binary crossentropy (see losses.py):
        # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

        # Calculate the binary crossentropy
        b_ce = K.binary_crossentropy(y_true, y_pred)

        # Apply the weights
        weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return K.mean(weighted_b_ce)

    return weighted_binary_crossentropy

csv.field_size_limit(sys.maxsize)

class UBCNN(object):

    def __init__(self):
        print("creating UBCNN ... ")
        print("... UBCNN created!")

    #Default net is ThomasNet_Stent:
    def compile_model(self,input_shape=(128,128),n_target_feat=3):

        I,J = input_shape

        net = Sequential()
        net.add(Conv2D(20, (5, 5), input_shape=(I, J, 1)))
        net.add(Activation('relu'))
        net.add(MaxPooling2D(pool_size=(14, 14)))

        #net.add(Conv2D(10, (3, 3)))
        #net.add(Activation('relu'))
        #net.add(MaxPooling2D(pool_size=(2, 2)))
        #net.add(Dropout(0.25))#ME

        net.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        net.add(Dense(10))
        net.add(Activation('relu'))
        net.add(Dropout(0.5))
        net.add(Dense(n_target_feat))
        net.add(Activation('sigmoid'))

        print("compiling model ... ")

        optimizer = Adam()

        net.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        print("... compiled! (details below)")

        self.model = net
        self.trained = False
        net.summary()

    def train_model(self,X_train, y_train, X_val=[], y_val=[], batch_size=16, n_epochs=1, save_name=None, callbacks=[]):

        net = self.model
        history = None
        print("training model ...")
        if len(X_val)!=0 and len(y_val)!=0:
            history = net.fit(X_train,y_train,epochs=n_epochs,batch_size=batch_size,verbose=2,callbacks=callbacks,validation_data=(X_val, y_val))
        else:
            history = net.fit(X_train,y_train,epochs=n_epochs,batch_size=batch_size,verbose=2,callbacks=callbacks)

        net.trained = True

        if save_name != None:
            net.save(save_name)

        return history

    def train_model_DA(self,X_train, y_train, X_val=[], y_val=[], batch_size=16, n_epochs=1, save_name=None, callbacks=[]):
	datagen = ImageDataGenerator(
	    	    featurewise_center=False,  # set input mean to 0 over the dataset
	    	    samplewise_center=False,  # set each sample mean to 0
	    	    featurewise_std_normalization=False,  # divide inputs by std of the dataset
	    	    samplewise_std_normalization=False,  # divide each input by its std
	    	    zca_whitening=False,  # apply ZCA whitening
	    	    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
	    	    width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
	    	    height_shift_range=0,  # randomly shift images vertically (fraction of total height)
	    	    horizontal_flip=True,  # randomly flip images
	    	    vertical_flip=True)  # randomly flip images

    	datagen.fit(X_train)

        net = self.model
        history = None
        print("... training model with RealTime Data Augmentation ...")

	#Start Data Fitting With Augmented DataSet

    	history = net.fit_generator(datagen.flow(X_train, y_train,
                                         batch_size=batch_size),
                            epochs=n_epochs,
                            validation_data=(X_val, y_val),
                            workers=4,callbacks=callbacks)

        net.trained = True

        if save_name != None:
            net.save(save_name)
	return history
    def save(self,model_path):

        net = self.model
        net.save(model_path)

    def load_model(self,filename):

        net = load_model(filename)
        self.model = net
        net.summary()
        return self.model

    def predict(self,X):

        net = self.model
        out_net = net.predict(X)
        y_pred = np.array([val for sublist in out_net for val in sublist])
        print(y_pred.shape)
        print("predicted ...")
        return y_pred

    def test_model(self,X_test,y_test):
        net = self.model
        out_net = net.predict(X_test)
        y_pred = np.array([val for sublist in out_net for val in sublist])
        y_pred[y_pred>=0.5] = 1.0
        y_pred[y_pred<0.5] = 0.0
        test_error = np.sum(np.abs(y_test-y_pred))/float(len(y_test))
        print(y_pred.shape)
        print(y_test.shape)
        print(test_error)
        print("--------")
        return test_error

    def evaluate(self,X,y):
        net = self.model
        return net.evaluate(X,y)


class UBCNN_MIX(UBCNN):

    def __init__(self,input_channels=1):
        self.input_channels=input_channels
        print("creating UBCNN MIXTURE... ")
        print("... UBCNN MIXTURE created!")

    #Default net is ThomasNet_Stent:
    def compile_model(self,input_shape=(128,128),n_target_feat=3):

        I,J = input_shape
        input_channels = self.input_channels
        #STENT
        stent = Sequential()
        stent.add(Conv2D(20, (5, 5), input_shape=(I, J, input_channels)))
        stent.add(Activation('relu'))
        stent.add(MaxPooling2D(pool_size=(14, 14)))
        stent.add(Flatten())

        #FIBRO
        fibro = Sequential()
        fibro.add(Conv2D(10, (4, 4), input_shape=(I, J, input_channels)))
        fibro.add(Activation('relu'))
        fibro.add(MaxPooling2D(pool_size=(14, 14),strides=(12, 12)))
        fibro.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

        #CALCIO
        calcio = Sequential()
        calcio.add(Conv2D(20, (10, 10), input_shape=(I, J, input_channels)))
        calcio.add(Activation('relu'))
        calcio.add(AveragePooling2D(pool_size=(14, 14)))
        calcio.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

        top = Concatenate()([stent.output,fibro.output,calcio.output])
        top = Dense(20, activation='relu')(top)
        top = Dropout(0.5)(top)
        top = Dense(n_target_feat, activation='sigmoid')(top)

        net = Model(inputs=[stent.input,fibro.input,calcio.input], outputs=top)

        print("compiling model ... ")

        optimizer = Adam()

        net.compile(loss='binary_crossentropy', optimizer=optimizer)

        print("... compiled! (details below)")

        self.model = net
        self.trained = False
        net.summary()

    def train_model(self,X_train, y_train, X_val=[], y_val=[], batch_size=16, n_epochs=1, save_name=None, callbacks=[]):

        return super(UBCNN_MIX, self).train_model([X_train,X_train,X_train],y_train,X_val=[X_val,X_val,X_val],y_val=y_val,batch_size=batch_size,n_epochs=n_epochs,save_name=save_name,callbacks=callbacks)

    def test_model(self,X_test,y_test):

        return super(UBCNN_MIX, self).test_model([X_test,X_test,X_test],y_test)


    def evaluate(self,X,y):

        return super(UBCNN_MIX, self).evaluate([X,X,X],y)


class ThomasNet_Stent(UBCNN):

    def __init__(self):
        print("creating ThomasNet for Stent classification ... ")
        print("... ThomasNet created!")

    def compile_model(self,input_shape=(128,128),n_target_feat=1):

        I,J = input_shape

        net = Sequential()
        net.add(Conv2D(20, (5, 5), input_shape=(I, J, 1)))
        net.add(Activation('relu'))
        net.add(MaxPooling2D(pool_size=(14, 14)))
        net.add(Dropout(0.25))#ME
        net.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

        net.add(Dense(n_target_feat))
        net.add(Activation('sigmoid'))

        print("compiling model ... ")

        optimizer = Adam()

        net.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        print("... compiled! (details below)")

        self.model = net
        self.trained = False
        net.summary()


class ThomasNet_Fibro(UBCNN):

    def __init__(self):
        print("creating ThomasNet for Fibro classification ... ")
        print("... ThomasNet created!")

    def compile_model(self,input_shape=(128,128),n_target_feat=1):

        I,J = input_shape

        net = Sequential()
        net.add(Conv2D(10, (4, 4), input_shape=(I, J, 1)))
        net.add(Activation('relu'))
        net.add(MaxPooling2D(pool_size=(14, 14),strides=(12, 12)))
        net.add(Dropout(0.25))#ME

        net.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

        net.add(Dense(n_target_feat))
        net.add(Activation('sigmoid'))

        print("compiling model ... ")

        optimizer = Adam()

        net.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        print("... compiled! (details below)")

        self.model = net
        self.trained = False
        net.summary()

class ThomasNet_Calcio(UBCNN):

    def __init__(self):
        print("creating ThomasNet for Calcio classification ... ")
        print("... ThomasNet created!")

    def compile_model(self,input_shape=(128,128),n_target_feat=1,size_conv_kernel=10,size_pool_kernel=14,n_neurons=20,nhidden=20,dropout_flag=1,dropout_par=0.5):

        I,J = input_shape

        net = Sequential()
        net.add(Conv2D(32, (10, 10), input_shape=(I, J, 1)))
        net.add(BatchNormalization())
        net.add(Activation('relu'))
        net.add(AveragePooling2D(pool_size=(14,14)))

        net.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

        net.add(Dense(n_target_feat))
        net.add(Activation('sigmoid'))

        print("compiling model ... ")

        optimizer = Adam()

        net.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        print("... compiled! (details below)")

        self.model = net
        self.trained = False
        net.summary()

class MauNet_Calcio(UBCNN):

    def __init__(self):
        print("creating MauNet for Calcio classification ... ")
        print("... MauNet created!")

    def compile_model(self,input_shape=(128,128),n_target_feat=1,nfilters=20,fsize=10):
        I,J = input_shape
        net = Sequential()

        net.add(Conv2D(32, (3, 3), input_shape=(I, J, 1), activation='relu'))
        net.add(MaxPooling2D(pool_size=(2, 2)))
        net.add(Dropout(0.25))

        net.add(Conv2D(64, (3, 3),activation='relu'))
        net.add(MaxPooling2D(pool_size=(2, 2)))
        net.add(Dropout(0.25))

        net.add(Conv2D(64, (3, 3), activation='relu'))
        net.add(MaxPooling2D(pool_size=(2, 2)))
        net.add(Dropout(0.25))

        # net.add(Conv2D(64, (3, 3), activation='relu',padding='same' ))
        # net.add(MaxPooling2D(pool_size=(2, 2)))
        # net.add(Dropout(0.4))

        net.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        net.add(Dense(512, activation='relu'))
        net.add(Dropout(0.25))
        net.add(Dense(1, activation='sigmoid'))
        net.summary()
        net.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        print("... compiled! (details below)")

        self.model = net
        self.trained = False
        net.summary()
class MauNet_Calcio_2L(UBCNN):

    def __init__(self):
        print("creating MauNet 2L for Calcio classification ... ")
        print("... MauNet created!")

    def compile_model(self,input_shape=(128,128),n_target_feat=1,nfilters=20,fsize=10,dropout=0.4):
        I,J = input_shape

        net = Sequential()
        net.add(Conv2D(64, (3, 3), input_shape=(I, J, 1)))
        net.add(Activation('relu'))
        net.add(MaxPooling2D(pool_size=(2, 2)))
        net.add(Dropout(0.1))

        net.add(Conv2D(32, (3, 3)))
        net.add(Activation('relu'))
        net.add(MaxPooling2D(pool_size=(2, 2)))
        net.add(Dropout(0.1))

        net.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

        net.add(Dense(n_target_feat))
        net.add(Activation('sigmoid'))

        print("compiling model ... ")

        optimizer = Adam()

        net.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        print("... compiled! (details below)")

        self.model = net
        self.trained = False
        net.summary()
class MauNet_Calcio_5L_BN(UBCNN):

    def __init__(self):
        print("creating MauNet 3L with BN for Calcio classification ... ")
        print("... MauNet created!")

    def compile_model(self,input_shape=(128,128),n_target_feat=1,nfilters=20,fsize=10,dropout=0.4):
        I,J = input_shape
        net = Sequential()

        net.add(Conv2D(64, (3, 3), input_shape=(I, J, 1), activation='relu'))
        net.add(BatchNormalization())
        net.add(MaxPooling2D(pool_size=(2, 2)))
        net.add(Dropout(0.4))

        net.add(Conv2D(64, (3, 3), activation='relu'))
        net.add(BatchNormalization())
        net.add(MaxPooling2D(pool_size=(2, 2)))
        net.add(Dropout(0.4))

        net.add(Conv2D(32, (3, 3), activation='relu'))
        net.add(BatchNormalization())
        net.add(MaxPooling2D(pool_size=(2, 2)))
        net.add(Dropout(0.4))

        net.add(Conv2D(32, (3, 3), activation='relu'))
        net.add(BatchNormalization())
        net.add(MaxPooling2D(pool_size=(2, 2)))
        net.add(Dropout(0.4))

        net.add(Conv2D(16, (3, 3), activation='relu'))
        net.add(BatchNormalization())
        net.add(MaxPooling2D(pool_size=(2, 2)))
        net.add(Dropout(0.4))

        net.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        
        net.add(Dense(n_target_feat))
        net.add(Activation('sigmoid'))

        net.compile(loss='binary_crossentropy',
                      optimizer='Adam',
                      metrics=['accuracy'])

        print("... compiled! (details below)")

        self.model = net
        self.trained = False
        net.summary()

class MauNet_Calcio_3L(UBCNN):

    def __init__(self):
        print("creating MauNet 3L No BN for Calcio classification ... ")
        print("... MauNet created!")

    def compile_model(self,input_shape=(128,128),n_target_feat=1,nfilters=20,fsize=10,dropout=0.4):
        I,J = input_shape
        net = Sequential()

        net.add(Conv2D(nfilters, (3, 3), input_shape=(I, J, 1), activation='relu'))
        net.add(MaxPooling2D(pool_size=(2, 2)))
        net.add(Dropout(dropout))

        tpt = nfilters*2

        net.add(Conv2D(tpt, (3, 3), activation='relu'))
        net.add(MaxPooling2D(pool_size=(2, 2)))
        net.add(Dropout(dropout))

        net.add(Conv2D(tpt, (3, 3), activation='relu'))
        net.add(MaxPooling2D(pool_size=(2, 2)))
        net.add(Dropout(dropout))


        net.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        
        net.add(Dense(n_target_feat))
        net.add(Activation('sigmoid'))

        net.compile(loss='binary_crossentropy',
                      optimizer='Adam',
                      metrics=['accuracy'])

        print("... compiled! (details below)")

        self.model = net
        self.trained = False
        net.summary()
class MauNet_Calcio_3L_BN(UBCNN):

    def __init__(self):
        print("creating MauNet 3L with BN for Calcio classification ... ")
        print("... MauNet created!")

    def compile_model(self,input_shape=(128,128),n_target_feat=1,nfilters=20,fsize=10,dropout=0.4):
        I,J = input_shape
        net = Sequential()

        net.add(Conv2D(nfilters, (3, 3), input_shape=(I, J, 1), activation='relu'))
        net.add(BatchNormalization())
        net.add(MaxPooling2D(pool_size=(2, 2)))
        net.add(Dropout(dropout))

        tpt = nfilters*2

        net.add(Conv2D(tpt, (3, 3), activation='relu'))
        net.add(BatchNormalization())
        net.add(MaxPooling2D(pool_size=(2, 2)))
        net.add(Dropout(dropout))

        net.add(Conv2D(tpt, (3, 3), activation='relu'))
        net.add(BatchNormalization())
        net.add(MaxPooling2D(pool_size=(2, 2)))
        net.add(Dropout(dropout))


        net.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        
        net.add(Dense(n_target_feat))
        net.add(Activation('sigmoid'))

        net.compile(loss='binary_crossentropy',
                      optimizer='Adam',
                      metrics=['accuracy'])

        print("... compiled! (details below)")

        self.model = net
        self.trained = False
        net.summary()

class MauNet_Calcio_4L(UBCNN):

    def __init__(self):
        print("creating MauNet 4L No BN for Calcio classification ... ")
        print("... MauNet created!")

    def compile_model(self,input_shape=(128,128),n_target_feat=1,nfilters=20,fsize=10,dropout=0.4):
        I,J = input_shape
        net = Sequential()

        net.add(Conv2D(nfilters, (3, 3), input_shape=(I, J, 1), activation='relu'))
        net.add(MaxPooling2D(pool_size=(2, 2)))
        net.add(Dropout(dropout))

        net.add(Conv2D(nfilters, (3, 3),activation='relu'))
        net.add(MaxPooling2D(pool_size=(2, 2)))
        net.add(Dropout(dropout))

        tpt = nfilters*2

        net.add(Conv2D(tpt, (3, 3), activation='relu'))
        net.add(MaxPooling2D(pool_size=(2, 2)))
        net.add(Dropout(dropout))

        net.add(Conv2D(tpt, (3, 3), activation='relu'))
        net.add(MaxPooling2D(pool_size=(2, 2)))
        net.add(Dropout(dropout))

        net.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        
        net.add(Dense(n_target_feat))
        net.add(Activation('sigmoid'))

        net.compile(loss='binary_crossentropy',
                      optimizer='Adam',
                      metrics=['accuracy'])

        print("... compiled! (details below)")

        self.model = net
        self.trained = False
        net.summary()
class MauNet_Calcio_4L_BN(UBCNN):

    def __init__(self):
        print("creating MauNet 4L with BN for Calcio classification ... ")
        print("... MauNet created!")

    def compile_model(self,input_shape=(128,128),n_target_feat=1,nfilters=20,fsize=10,dropout=0.4):
        I,J = input_shape
        net = Sequential()

        net.add(Conv2D(nfilters, (3, 3), input_shape=(I, J, 1), activation='relu'))
        net.add(BatchNormalization())
        net.add(MaxPooling2D(pool_size=(2, 2)))
        net.add(Dropout(dropout))

        net.add(Conv2D(nfilters, (3, 3),activation='relu'))
        net.add(BatchNormalization())
        net.add(MaxPooling2D(pool_size=(2, 2)))
        net.add(Dropout(dropout))

        tpt = nfilters*2

        net.add(Conv2D(tpt, (3, 3), activation='relu'))
        net.add(BatchNormalization())
        net.add(MaxPooling2D(pool_size=(2, 2)))
        net.add(Dropout(dropout))

        net.add(Conv2D(tpt, (3, 3), activation='relu'))
        net.add(BatchNormalization())
        net.add(MaxPooling2D(pool_size=(2, 2)))
        net.add(Dropout(dropout))

        net.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        
        net.add(Dense(n_target_feat))
        net.add(Activation('sigmoid'))

        net.compile(loss='binary_crossentropy',
                      optimizer='Adam',
                      metrics=['accuracy'])

        print("... compiled! (details below)")

        self.model = net
        self.trained = False
        net.summary()

class ThomasNet_Bifurcation(UBCNN):

    def __init__(self):
        print("creating ThomasNet for Bifurcation classification ... ")
        print("... ThomasNet created!")

    def compile_model(self,input_shape=(128,128),n_target_feat=1,size_conv_kernel=10,size_pool_kernel=14,n_neurons=20,nhidden=20,dropout_flag=1,dropout_par=0.5):

        I,J = input_shape

        net = Sequential()
        net.add(Conv2D(20, (3, 3), input_shape=(I, J, 1)))
        net.add(Activation('relu'))
        net.add(MaxPooling2D(pool_size=(2,2)))

        net.add(Conv2D(40, (3, 3)))
        net.add(Activation('relu'))
        net.add(MaxPooling2D(pool_size=(2,2)))

        net.add(Conv2D(80, (3, 3)))
        net.add(Activation('relu'))
        net.add(MaxPooling2D(pool_size=(2,2)))

        net.add(Conv2D(120, (3, 3)))
        net.add(Activation('relu'))
        net.add(MaxPooling2D(pool_size=(2,2)))

        net.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

        net.add(Dense(n_target_feat))
        net.add(Activation('sigmoid'))

        print("compiling model ... ")

        optimizer = Adam()

        #loss=create_weighted_binary_crossentropy(0.05,0.95)
        net.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        print("... compiled! (details below)")

        self.model = net
        self.trained = False
        net.summary()

class RSNet_Calcio(UBCNN):

    def __init__(self):
        print("creating RSNet for Calcio classification ... ")
        print("... RSNet created!")

    def compile_model(self,input_shape=(128,128),n_target_feat=1,size_conv_kernel=10,size_pool_kernel=14,n_neurons=20,nhidden=20,dropout_flag=1,dropout_par=0.5):

        I,J = input_shape

        net = Sequential()
        net.add(Conv2D(20, (20, 20), padding='same', input_shape=(I, J, 1)))
        net.add(Activation('relu'))
        net.add(MaxPooling2D(pool_size=(20,20)))
        net.add(Dropout(0.25))#ME

        net.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

        net.add(Dense(nhidden))
        net.add(Activation('relu'))
        net.add(Dropout(dropout_par))#ME

        net.add(Dense(n_target_feat))
        net.add(Activation('sigmoid'))

        print("compiling model ... ")
        print("using convolution filters of size %dx%d"%(size_conv_kernel, size_conv_kernel))
        print("using pooling filters of size %dx%d"%(size_pool_kernel, size_pool_kernel))

        optimizer = Adam()

        net.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        print("... compiled! (details below)")

        self.model = net
        self.trained = False
        net.summary()

class RSNet_Bifurcation(UBCNN):

    def __init__(self):
        print("creating RSNet for Bifurcation classification ... ")
        print("... RSNet created!")

    def compile_model(self,input_shape=(128,128),n_target_feat=1,size_conv_kernel=10,size_pool_kernel=14,n_neurons=20,nhidden=20,dropout_flag=1,dropout_par=0.5):

        I,J = input_shape

        net = Sequential()
        net.add(Conv2D(20, (20, 20), padding='same', input_shape=(I, J, 1)))
        net.add(Activation('relu'))
        net.add(MaxPooling2D(pool_size=(20,20)))
        net.add(Dropout(0.25))#ME

        net.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

        net.add(Dense(nhidden))
        net.add(Activation('relu'))
        net.add(Dropout(dropout_par))#ME

        net.add(Dense(n_target_feat))
        net.add(Activation('sigmoid'))

        print("compiling model ... ")
        print("using convolution filters of size %dx%d"%(size_conv_kernel, size_conv_kernel))
        print("using pooling filters of size %dx%d"%(size_pool_kernel, size_pool_kernel))

        optimizer = Adam()

        net.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        print("... compiled! (details below)")

        self.model = net
        self.trained = False
        net.summary()

class DeepNet_Bifurcation(UBCNN):

    def __init__(self):
        print("creating DeepNet for Bifurcation classification ... ")
        print("... RSNet created!")

    def compile_model(self,input_shape=(128,128),n_target_feat=1,size_conv_kernel=10,size_pool_kernel=14,n_neurons=20,nhidden=20,dropout_flag=1,dropout_par=0.5):

        I,J = input_shape

        net = Sequential()
        net.add(Conv2D(32, (3, 3), padding='same', input_shape=(I, J, 1)))
        net.add(Activation('relu'))
        net.add(MaxPooling2D(pool_size=(2,2)))
        net.add(Dropout(0.25))#ME

        net.add(Conv2D(32, (3, 3), padding='same'))
        net.add(Activation('relu'))
        net.add(MaxPooling2D(pool_size=(2,2)))
        net.add(Dropout(0.25))#ME

        net.add(Conv2D(32, (3, 3), padding='same'))
        net.add(Activation('relu'))
        net.add(MaxPooling2D(pool_size=(2,2)))
        net.add(Dropout(0.25))#ME

        net.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

        net.add(Dense(nhidden))
        net.add(Activation('relu'))
        net.add(Dropout(dropout_par))#ME

        net.add(Dense(n_target_feat))
        net.add(Activation('sigmoid'))

        optimizer = Adam()

        net.compile(loss=create_weighted_binary_crossentropy(0.05,0.95), optimizer=optimizer, metrics=['accuracy'])

        print("... compiled!!! (details below)")

        self.model = net
        self.trained = False
        net.summary()


class ThomasNet_CalcioColor(UBCNN):

    def __init__(self):
        print("creating ThomasNet for Calcio classification ... ")
        print("... ThomasNet created!")

    def compile_model(self,input_shape=(128,128),n_target_feat=1):

        I,J = input_shape

        net = Sequential()
        net.add(Conv2D(20, (10, 10), input_shape=(I, J, 3)))
        net.add(Activation('relu'))
        net.add(AveragePooling2D(pool_size=(14, 14)))

        net.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

        net.add(Dense(n_target_feat))
        net.add(Activation('sigmoid'))

        print("compiling model ... ")

        optimizer = Adam()

        net.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        print("... compiled! (details below)")

        self.model = net
        self.trained = False
        net.summary()

class ImageReader:

    def __init__(self):
        print("creating ImageReader ... ")
        print("... Reader created!")

    def read_gabriel_color(self,main_folder="/",label='Calcio',gray=False):

        files_yes = []
        files_no = []

        I,J,K = 128,128,3

        if gray:
            K=1

        n_features = 1

        if label == 'Calcio':

            calci_yes_subdir = 'Calci/SiC'
            calci_no_subdir = 'Calci/NoC'

            subdir = os.path.join(main_folder,calci_yes_subdir)
            file_list = sorted(os.listdir(subdir))
            files_yes = [os.path.join(subdir, f) for f in file_list if os.path.isfile(os.path.join(subdir, f))]

            subdir = os.path.join(main_folder,calci_no_subdir)
            file_list = sorted(os.listdir(subdir))
            files_no = [os.path.join(subdir, f) for f in file_list if os.path.isfile(os.path.join(subdir, f))]

            print("Loading Calcio Data")

        elif label == 'Stent':

            stent_yes_subdir = 'Stent/SiS'
            stent_no_subdir = 'Stent/NoS'

            subdir = os.path.join(main_folder,stent_yes_subdir)
            file_list = sorted(os.listdir(subdir))
            files_yes = [os.path.join(subdir, f) for f in file_list if os.path.isfile(os.path.join(subdir, f))]

            subdir = os.path.join(main_folder,stent_no_subdir)
            file_list = sorted(os.listdir(subdir))
            files_no = [os.path.join(subdir, f) for f in file_list if os.path.isfile(os.path.join(subdir, f))]

        else:

            fibro_yes_subdir = 'Fibro/SiF'
            fibro_no_subdir = 'Fibro/NoF'

            subdir = os.path.join(main_folder,fibro_yes_subdir)
            file_list = sorted(os.listdir(subdir))
            files_yes = [os.path.join(subdir, f) for f in file_list if os.path.isfile(os.path.join(subdir, f))]

            subdir = os.path.join(main_folder,fibro_no_subdir)
            file_list = sorted(os.listdir(subdir))
            files_no = [os.path.join(subdir, f) for f in file_list if os.path.isfile(os.path.join(subdir, f))]


        N = len(files_yes)+len(files_no)

        X = np.zeros((N,I,J,K),dtype=np.float32)
        Y = np.zeros((N,n_features),dtype=np.int32)

        count_found = 0
        for f in files_yes:
            img_path = os.path.abspath(f)

            if gray:
                img=image.load_img(img_path,grayscale=True) #target_size=(224, 224)
            else:#color
                img = image.load_img(img_path)

            x = image.img_to_array(img)
            X[count_found,:,:,:] = x
            Y[count_found] = 1
            count_found += 1

        for f in files_no:
            img_path = os.path.abspath(f)

            if gray:
                img=image.load_img(img_path,grayscale=True) #target_size=(224, 224)
            else:#color
                img = image.load_img(img_path)

            x = image.img_to_array(img)
            X[count_found,:,:,:] = x
            Y[count_found] = 0
            count_found += 1

        print(X.shape)
        print(Y.shape)

        return X,Y

    def read_data(self,main_folder='DICOM_FRA',filename_labels='labels.txt',n_features=3):

        #READ FRAMES
        subdirs = os.listdir(main_folder)
        subdirs.sort()

        count_frames = 0
        files = []
        for dir_ in subdirs:
            subdir = os.path.join(main_folder,dir_)
            if os.path.isdir(subdir):
                files_ = [os.path.join(subdir, f) for f in os.listdir(subdir) if os.path.isfile(os.path.join(subdir, f))]
                for f in files_:
                    files.append(f)
                    count_frames+=1

        count_labels = 0
        n_features = 3

        #READ LABELS
        print("reading labels ...")
        with open(filename_labels, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)
            frame_labels = dict()
            for row in reader:
                count_labels +=1
                name_elements = row[0].split('\\',2)
                frame_full_name = str(name_elements[-1])
                frame_full_name = frame_full_name.replace('\\','/')
                labels = np.array([int(i) for i in row[1:n_features+1]])
                frame_labels[frame_full_name] = labels


        n_frames_labelled = count_labels
        n_frames_unlabelled = count_frames - count_labels


        count_found = 0
        for f in files:
            if f in frame_labels:
                count_found+=1

        I=512
        J=512
        K=1

        X = np.zeros((count_found,I,J,K),dtype=np.float32)
        Y = np.zeros((count_found,n_features),dtype=np.int32)

        count_found = 0
        it = 0
        it_lim = 1000000000
        for f in files:
            if f in frame_labels:
                img_path = os.path.abspath(f)
                img = image.load_img(img_path,grayscale=True) #target_size=(224, 224)
                x = image.img_to_array(img)
                X[count_found,:,:,:] = x
                Y[count_found,:] = frame_labels[f]
                count_found+=1
            it+=1
            if it > it_lim:
                break

        print("FOUND %d of %d"%(count_found,n_frames_labelled))
        print("SHAPE X: ",X.shape)
        print("SHAPE Y: ",Y.shape)

        nnz0 = np.count_nonzero(Y[:,0])
        nnz1 = np.count_nonzero(Y[:,1])
        nnz2 = np.count_nonzero(Y[:,2])

        print("NON ZERO COLUMN 0=%d, rate %f"%(nnz0,float(nnz0)/float(len(Y[:,0]))))
        print("NON ZERO COLUMN 1=%d, rate %f"%(nnz1,float(nnz1)/float(len(Y[:,1]))))
        print("NON ZERO COLUMN 2=%d, rate %f"%(nnz2,float(nnz2)/float(len(Y[:,2]))))

        return X,Y

    def read_pullbacks_from_CSV(self,names='PULLBACKS_NAMES.csv',file_x='PULLBACKS_X.csv',file_y='PULLBACKS_Y.csv',n_features=3,dim=128,return_names=False):

        X,Y = self.read_fromCSV(csv_x=file_x,csv_y=file_y,dim=dim,n_features=n_features)

        start_indices = []
        end_indices = []

        pullback_names = []

        with open(names,'r') as names_file:
            for names in names_file:
                pullback_name = names.split(",")[0]
                frame_name = names.split(",")[1]
                pullback_names.append(pullback_name)

        last_name = pullback_names[0]
        start_indices.append(0)
        for idx in range(1,len(pullback_names)):
            current_name = pullback_names[idx]
            if current_name != last_name:
                start_indices.append(int(idx))
                last_name = current_name
                end_indices.append(int(idx-1))

        end_indices.append(int(len(pullback_names)-1))

        if return_names is False:
            return X,Y,start_indices,end_indices

        return X,Y,start_indices,end_indices,pullback_names


    def read_fromCSV(self,csv_x='X_DICOM_FRA.csv',csv_y='Y_DICOM_FRA.csv',dim=120,n_features=3):

        counter = 0

        with open(csv_x, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            for row in reader:
                counter +=1

        n_data = counter
        X = np.zeros((n_data,dim,dim,1),dtype=np.float32)
        Y = np.zeros((n_data,n_features),dtype=np.float32)

        counter = 0
        with open(csv_x, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            for row in reader:
                pattern = np.reshape(np.array(row,dtype=np.float32),(dim,dim))
                X[counter,:,:,0] = pattern
                counter +=1

        counter = 0
        with open(csv_y, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            for row in reader:
                Y[counter,:] = np.array([float(i) for i in row])
                counter +=1

        print(X.shape)
        print(Y.shape)

        return X,Y


class ExperimentParams:

    def __init__(self,net_architecture,shape_X_train,shape_Y_train,target_vars,nepochs):
        self.net_architecture = net_architecture
        self.shape_X_train = shape_X_train
        self.shape_Y_train = shape_X_train
        self.target_vars = target_vars
        self.nepochs = nepochs

class ExperimentResults:

        def __init__(self,train_accuracy,test_accuracy,history_tr,history_ts,others=None):
            self.train_accuracy = train_accuracy
            self.test_accuracy = test_accuracy
            self.history_tr=history_tr
            self.history_ts=history_ts
            self.others = others

class Pullback(object):

    def __init__(self, X, name):
        #X: data matrix
        #X shape (N,I,J,1) N: number of frames in the pullback
        #I: horizontal resolution, J: vertical resolution
        #name: string label to identify the pullback and match with the ground truth

        self.X = X
        self.name = name



class Deep_CholletNet(UBCNN):

    def __init__(self):
        print("creating Deep Net ... ")
        print("... DeepNet created!")

    def compile_model(self,input_shape=(128,128),n_target_feat=1):

        I,J = input_shape

        net = Sequential()
        net.add(Conv2D(32, (3, 3), input_shape=(I, J, 1)))
        net.add(Activation('relu'))
        net.add(MaxPooling2D(pool_size=(2, 2)))
        net.add(Dropout(0.25))#ME

        net.add(Conv2D(32, (3, 3)))
        net.add(Activation('relu'))
        net.add(MaxPooling2D(pool_size=(2, 2)))
        net.add(Dropout(0.25))#ME

        net.add(Conv2D(32, (3, 3)))
        net.add(Activation('relu'))
        net.add(MaxPooling2D(pool_size=(2, 2)))
        net.add(Dropout(0.25))#ME

        net.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        net.add(Dense(32))
        net.add(Activation('relu'))
        net.add(Dropout(0.5))
        net.add(Dense(n_target_feat))
        net.add(Activation('sigmoid'))

        print("compiling model ... ")

        optimizer = Adam()

        net.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        print("... compiled! (details below)")

        self.model = net
        self.trained = False
        net.summary()

class Deep_MultiLabelNet(UBCNN):

    def __init__(self):
        print("creating Deep Net ... ")
        print("... DeepNet created!")

    def test_model(self,X_test,Ytest):

        net = self.model
        out_net = net.predict(X_test)

        print "predicting ..."
        print out_net.shape

        I,K = Ytest.shape

        accuracies = np.zeros(K)

        for k in range(K):
            hits = 0
            cases = 0
            for i in range(I):
                if (out_net[i,k] >= 0.5) and (Ytest[i,k] >= 0.5):
                    hits +=1
                else:
                    if((out_net[i,k] < 0.5) and (Ytest[i,k] < 0.5)):
                        hits +=1
                cases +=1
            acc = float(hits)/float(cases)
            accuracies[k] = 1.0-acc

        return accuracies

    def predict(self,X):

        net = self.model
        out_net = net.predict(X)
        print(out_net.shape)
        print("predicted ...")
        return out_net

    def compile_model(self,input_shape=(128,128),n_target_feat=3):

        I,J = input_shape

        net = Sequential()
        net.add(Conv2D(32, (3, 3), input_shape=(I, J, 1)))
        net.add(Activation('relu'))
        net.add(MaxPooling2D(pool_size=(2, 2)))
        net.add(Dropout(0.25))#ME

        net.add(Conv2D(32, (3, 3)))
        net.add(Activation('relu'))
        net.add(MaxPooling2D(pool_size=(2, 2)))
        net.add(Dropout(0.25))#ME

        net.add(Conv2D(32, (3, 3)))
        net.add(Activation('relu'))
        net.add(MaxPooling2D(pool_size=(2, 2)))
        net.add(Dropout(0.25))#ME

        net.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        net.add(Dense(32))
        net.add(Activation('relu'))
        net.add(Dropout(0.5))
        net.add(Dense(n_target_feat))
        net.add(Activation('sigmoid'))

        print("compiling model ... ")

        optimizer = Adam()

        net.compile(loss='binary_crossentropy', optimizer=optimizer)

        print("... compiled! (details below)")

        self.model = net
        self.trained = False
        net.summary()

class ColorMap(object):

    def __init__(self,colormapfile="colormap.txt"):
        print("creating ColorMap ... ")
        self.colormap = self.read_colormap(filename=colormapfile)
        print("... Jet Colormap loaded!")


    def read_colormap(self, filename="colormap.txt"):

        counter = 0
        csv_x = filename
        jet_map = np.zeros((256,3),dtype=np.float32)

        with open(csv_x, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ',skipinitialspace=True)
            for row in reader:
                rgb = np.asarray([float(val) for val in row])
                jet_map[counter,:] = rgb
                counter+=1
        return jet_map

    def gray_to_RGB(self,X,normalized=True):

        N,I,J,K = X.shape

        K_new = 3
        X_new = np.zeros((N,I,J,K_new),dtype=np.float32)

        for n in range(N):
            for i in range(I):
                for j in range(J):
                    for k in range(K_new):
                        if normalized:
                            X_new[n,i,j,k] = self.colormap[int(round(255*X[n,i,j,0])),k]
                        else:
                            X_new[n,i,j,k] = self.colormap[int(round(X[n,i,j,0])),k]

        return X_new
