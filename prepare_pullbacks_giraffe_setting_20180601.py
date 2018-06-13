from keras.preprocessing.image import load_img
from keras.preprocessing import image
from skimage.transform import resize

import csv
import os
import sys
import numpy as np 

name_x = 'PULLBACKS_X_GIRAFFE.csv'
name_y = 'PULLBACKS_Y_GIRAFFE.csv'
name_names = 'PULLBACKS_NAMES_GIRAFFE.csv'
name_file_labels = 'clean_labels_giraffe.txt'

#read folder with images, eventually reduce resolution and save results to CSV
#in addition it saves labels and names of pullbacks and frames in the same order as images have been written
def prepare_pullbacks(input_folder='DICOM_FRA',filename_labels='labels.txt',n_features=3, pixels_in=512, pixels_out=128):

    #READ FRAMES 
    subdirs = os.listdir(input_folder)
    subdirs.sort()

    count_frames = 0
    files = []
    for dir_ in subdirs:
        subdir = os.path.join(input_folder,dir_)
        if os.path.isdir(subdir):
            files_ = [os.path.join(subdir, f) for f in os.listdir(subdir) if os.path.isfile(os.path.join(subdir, f))]
            files_.sort()
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

    K=1

    count_found = 0
    for f in files:
        print f
        if f in frame_labels:
            count_found+=1

    print("LABEL FILE SIZE:%d"%count_labels)
    print("LABELS FOUND:%d"%count_found)
    print("FRAMES FOUND:%d"%count_frames)


    X = np.zeros((count_found,pixels_out*pixels_out*K),dtype=np.float32)
    Y = np.zeros((count_found,n_features),dtype=np.int32)
    names = []

    count_found = 0
    it=0
    it_lim=1000000000

    names_file = open(name_names,'w')

    for f in files:
        if f in frame_labels:
            img_path = os.path.abspath(f)
            img = image.load_img(img_path,grayscale=True) #target_size=(224, 224)
            x = image.img_to_array(img)
            
            if pixels_in != pixels_out:
                x = x/255.0
                new_im = np.reshape(x,(pixels_in,pixels_in))
                new_im_small = resize(new_im, (pixels_out,pixels_out), order=1, preserve_range=True)
                new_im_small = np.reshape(new_im_small,pixels_out*pixels_out)
                print "Resizing image %dx%d to %dx%d"%(pixels_in,pixels_in,pixels_out,pixels_out)
                x = 255.0*new_im_small
            else:
                x = np.reshape(x,pixels_out*pixels_out)

            X[count_found,:] = x
            Y[count_found,:] = frame_labels[f]
            
            elements_name = f.split('/',1)
            elements_name = elements_name[1].split('/',1)
            pullback_name = elements_name[0]
            frame_name = elements_name[1]
            names_file.write("%s,%s\n"%(pullback_name,frame_name))
            count_found+=1

        it+=1

        if it > it_lim:
            break
    
    names_file.close()
    np.savetxt(name_x, X)
    np.savetxt(name_y, Y, fmt='%d')
    print(X.shape)
    print(Y.shape)
    return X,Y

prepare_pullbacks(input_folder='DICOM_FRA',filename_labels=name_file_labels,pixels_in=512, pixels_out=128)

