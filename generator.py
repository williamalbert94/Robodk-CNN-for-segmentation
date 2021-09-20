# -*- coding: utf-8 -*-
import numpy as np
import tensorflow.keras
from tensorflow.keras.utils import to_categorical
import cv2
from random import shuffle
import os
from os.path import join, exists, dirname
import numpy as np
import matplotlib.pyplot as plt
import numpy
import scipy
from scipy import ndimage, misc
from scipy.ndimage.interpolation import zoom
from random import shuffle,choice
import random
from tensorflow.keras.utils import Sequence
from PIL import Image,ImageStat 
import pandas as pd
import keras
Image.MAX_IMAGE_PIXELS = 933120000

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=2, dim=(128,128), n_channels=1,
                 n_classes=10,norm="min_max",transformations = None, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.norm = norm
        self.transformations = transformations
        self.shuffle = shuffle
        self.global_index = 0
        self.gets_images()
        self.on_epoch_end()
        if norm=='ALL':
            self.trans_custom = ['Normal','flip_x','flip_y']
        else:
            self.trans_custom = ['Normal']
        self.load_data(path_image =list_IDs[0:1][0])

    def apply_some_transformations(self,img,mask=None):
        'Random morphological transformations'
        data = {"image": img, "mask": mask}
        augmented = self.transformations(**data)
        img_trans, mask_trans= augmented["image"], augmented["mask"]
        x=np.stack((img_trans,)*self.n_channels, axis=-1)
        if mask is None:
            return x
        else:
            y = np.stack((mask_trans,)*1, axis=-1)
            return x,y

    def gets_images(self):
        self.ids_images = np.unique(np.asarray(self.list_IDs))

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        #self.global_index +=1 
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(list_IDs_temp)
        return  x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def load_data(self,path_image):
        self.path_image = path_image
        self.path_mask = path_image.replace('Images','Mask')
        #print(self.path_mask)

        self.images=cv2.imread(self.path_image, cv2.IMREAD_ANYDEPTH)
        #print(self.images.shape)
        self.size_img = self.images.shape
        #self.images = Image.fromarray(self.images)
        #self.images = self.images.convert("L")  

        self.mask = cv2.imread(self.path_mask, 1)
        #self.mask = Image.fromarray(self.mask)
        #self.mask = self.mask.convert("L")





    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Generate data
        x = None
        y = None
        for i, ID in enumerate(list_IDs_temp):
            # Store sample    
            #print(ID)       
            if self.path_image!=ID:
                self.load_data(path_image = ID)
            image = self.images
            mask= self.mask
            mask = mask[:,:,2]
            #mask[mask==0]=13
            mask[mask==13]=11

            image = image/255

            selected_trans = random.choice(self.trans_custom)
            if selected_trans=='Normal':
                image=image
                mask=mask
            if selected_trans=='flip_x':
                image=np.flipud(image)
                mask=np.flipud(mask)
            if selected_trans=='flip_y':
                image=np.fliplr(image)
                mask=np.fliplr(mask)
            image=np.stack((image,)*self.n_channels, axis=-1)

            mask=np.stack((mask,)*1, axis=-1)
            #print(mask.shape)
            image = cv2.resize(image, (512,512) , interpolation = cv2.INTER_AREA)            
            mask = cv2.resize(mask, (512,512) , interpolation = cv2.INTER_AREA)


            image = np.expand_dims(image,axis=0)
            mask = np.expand_dims(mask,axis=0)

            if x is None :
                x = image

            else:
                x = np.concatenate((x,image),axis=0)
   

            if y is None :   
                y = mask  

            else:
                y = np.concatenate((y,mask),axis=0)          

        return x,to_categorical(y,self.n_classes)
