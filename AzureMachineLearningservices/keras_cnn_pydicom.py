from __future__ import print_function
import argparse
import pydicom
from matplotlib import pyplot, cm
import os
import sys
import numpy as np
import pandas as pd
import scipy
import keras
from keras.models import Sequential
from keras.layers import AveragePooling2D , Convolution2D , Flatten ,Dense, MaxPooling2D, Conv2D
from keras.preprocessing import utils
from keras.preprocessing.image import ImageDataGenerator


def get_data(dicom_dir):
    #resize the image to desired resolution
    #print("dicom_dir",os.listdir(dicom_dir), dicom_dir)
    xsize = 256; ysize = 256
    
    data = np.zeros((xsize, ysize, 100))
    #print("dicom_dir",os.listdir(dicom_dir), dicom_dir)
    for i, s in enumerate(os.listdir(dicom_dir)):
    
        img = np.array(pydicom.read_file(dicom_dir+ s).pixel_array)
        xscale = xsize/img.shape[0]
        yscale = ysize/img.shape[1]
        data[:,:,i] = scipy.ndimage.interpolation.zoom(img, [xscale, yscale])
    #returning a numpy array of shape 100,256,256
    return data

if __name__=='__main__': 
    
    parser=argparse.ArgumentParser()
    parser.add_argument('-i','--data',help='directory of where dicom files exists' )
    parser.add_argument('--epoch', help='how many epoch to train on')
    #parser.add_argument('--reload', help='path to where you save the previous model and use it to continue training')
    parser.add_argument('--save_model', help='path to where you want to save the model')
    args=parser.parse_args()
    os.makedirs(args.data,exist_ok=True)#tip1 - makedirs , since the mounting is not a physical mounting , you need to make sure you create the relative directory from within python to make it work
    print(os.path.expandvars(args.data))
    
    X=get_data(args.data+'/ChestCTscan/dicom/')#tip2 -if you have a sub directory from within the mounting point , you will need to specify that by hard-coded path
    X=np.moveaxis(X, -1, 0)
    print("check the dicom file shape should be 100, 256,256", X.shape)
    
    ### get label 1=contrast / 0=no contrast 
    df=pd.read_csv(args.data +'/ChestCTscan/overview.csv',encoding='utf-8',sep=',')
    del df['Unnamed: 0']
    y=df.iloc[:,1].values
    y= np.array([1 if yi else 0 for yi in y])
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train,y_test=train_test_split(X,y , test_size=0.1 , random_state=0)
    ## need to add a fake dimension in the end since keras image_generator expect 4 dim 
    X_train=np.expand_dims(X_train, axis=3)
    X_test=np.expand_dims(X_test,axis=3)
    X=np.expand_dims(X,axis=3)
    X_train.shape, X_test.shape , y_train.shape, y_test.shape
    
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

    test_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
    train_datagen.fit(X_train, augment=True, seed=123)
    test_datagen.fit(X_test, augment=True, seed=123)
    train_batch=train_datagen.flow(X, y, batch_size=20, seed=123, shuffle=True )
    test_batch=test_datagen.flow(X_test, y_test, batch_size=20, seed=123, shuffle=True )
    # Initialising the CNN
    classifier = Sequential()
    
    # Step 1 - Convolution
    classifier.add(Conv2D(32, (3, 3), input_shape = (256, 256 ,1), activation = 'relu', padding="same"))
    
    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
    # Adding a second convolutional layer
    classifier.add(Conv2D(64, 3, 3, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
    # Adding a second convolutional layer
    classifier.add(Conv2D(128, 3, 3, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
        
    # Adding a second convolutional layer
    classifier.add(Conv2D(256, 3, 3, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
    # Step 3 - Flattening
    classifier.add(Flatten())
    
    # Step 4 - Full connection
    classifier.add(Dense(units = 128, activation = 'relu')) # the output_dim is chosen by experience
    classifier.add(Dense(units = 1, activation = 'sigmoid'))
    
    # Compiling the CNN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    classifier.fit_generator(train_batch,
                             steps_per_epoch = 100,
                             nb_epoch = int(args.epoch),
                             validation_data = test_batch,
                             validation_steps = 25)
    from keras.models import load_model
    os.makedirs(args.save_model,exist_ok=True)
    # Creates a HDF5 file 'my_model.h5'
    classifier.save(args.save_model+'/ChestCTscan_epoch{}.h5'.format(args.epoch))
    
    
    # Returns a compiled model identical to the previous one
    #model = load_model(save_model+'ChestCTscan.h5')
    

    
