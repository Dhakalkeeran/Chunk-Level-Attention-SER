#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: winston lin
"""
import numpy as np
from scipy.io import loadmat
import keras
import tensorflow
import random
from utils import getPaths, DynamicChunkSplitTrainingData
# Ignore warnings & Fix random seed
import warnings
warnings.filterwarnings("ignore")
random.seed(999)
random_seed=99

class DataGenerator_LLD(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, root_dir, label_dir, batch_size, split_set, shuffle=True):
        'Initialization'
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.batch_size = batch_size
        self.split_set = split_set                        # 'Train' or 'Validation'
        self.shuffle = shuffle
        # Loading Norm-Feature Parameters
        self.Feat_mean = loadmat('./NormTerm/feat_norm_means.mat')['normal_para']
        self.Feat_std = loadmat('./NormTerm/feat_norm_stds.mat')['normal_para']  
        # Loading Norm-Label Parameters
        self.Label_mean_act = loadmat('./NormTerm/act_norm_means.mat')['normal_para'][0][0]
        self.Label_std_act = loadmat('./NormTerm/act_norm_stds.mat')['normal_para'][0][0]
        self.Label_mean_dom = loadmat('./NormTerm/dom_norm_means.mat')['normal_para'][0][0]
        self.Label_std_dom = loadmat('./NormTerm/dom_norm_stds.mat')['normal_para'][0][0]
        self.Label_mean_val = loadmat('./NormTerm/val_norm_means.mat')['normal_para'][0][0]
        self.Label_std_val = loadmat('./NormTerm/val_norm_stds.mat')['normal_para'][0][0]         
        # Loading Data Paths/Labels
        self._paths, self._labels = getPaths(label_dir, split_set)
        self._paths, self._labels = self.check_valid_data(self._paths, self._labels)
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(getPaths(self.label_dir, self.split_set)[0])/self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find Batch list of Loading Paths
        list_paths_temp = [self._paths[k] for k in indexes]
        list_labels_temp = [self._labels[k] for k in indexes]
        
        # Generate data
        data, label = self.__data_generation(list_paths_temp, list_labels_temp)
        return data, label
        
    def check_valid_data(self, paths, labels):
        "Checks if data is valid which is the length of data greater or equal to 62"
        new_paths = []
        new_labels = np.array([])
        for i in range(len(paths)):
            # Store Norm-Data
            x = loadmat(self.root_dir + paths[i].replace('.wav','.mat'))['Audio_data']
            x = x[:,1:]                                     # remove time-info from the extracted OpenSmile LLDs
            x = (x-self.Feat_mean)/self.Feat_std            # LLDs feature normalization (z-norm)
            # Bounded NormFeat Ranging from -3~3 and assign NaN to 0
            x[np.isnan(x)]=0
            x[x>3]=3
            x[x<-3]=-3
            if np.array(x).shape[0] < 62:
                continue
            new_paths.append(paths[i])
            if i == 0:
                new_labels = np.concatenate((new_labels, labels[i]))
            else:
                new_labels = np.row_stack((new_labels, labels[i]))
        return new_paths, new_labels

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        _paths, _labels = getPaths(self.label_dir, self.split_set)
        _paths, _labels = self.check_valid_data(_paths, _labels)
        self.indexes = np.arange(len(_paths))
        if self.shuffle == True:
            np.random.seed(random_seed)
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_paths_temp, list_labels_temp):
        'Generates data containing batch_size with fixed chunck samples'           
        batch_x = []
        batch_y = []
        # print(np.array(list_paths_temp).shape)
        # print(np.array(list_labels_temp).shape)
        for i in range(len(list_paths_temp)):
            # Store Norm-Data
            x = loadmat(self.root_dir + list_paths_temp[i].replace('.wav','.mat'))['Audio_data']
            # we use the Interspeech 2013 computational paralinguistics challenge LLDs feature set
            # which includes totally 130 features (i.e., the "IS13_ComParE" configuration)
            x = x[:,1:]                                     # remove time-info from the extracted OpenSmile LLDs
            x = (x-self.Feat_mean)/self.Feat_std            # LLDs feature normalization (z-norm)
            # Bounded NormFeat Ranging from -3~3 and assign NaN to 0
            x[np.isnan(x)]=0
            x[x>3]=3
            x[x<-3]=-3            
            # Store Norm-Label
            y_act = (list_labels_temp[i][0]-self.Label_mean_act)/self.Label_std_act
            y_dom = (list_labels_temp[i][1]-self.Label_mean_dom)/self.Label_std_dom
            y_val = (list_labels_temp[i][2]-self.Label_mean_val)/self.Label_std_val
            # if np.array(x).shape[0] < 62:
            #     print(f"Inside Data generation, x: {np.array(x).shape}")
            #     print(f"Inside Data generation, y: {np.array(y).shape}")
            #     continue
            batch_x.append(x)
            batch_y.append((y_act, y_dom, y_val))

        # split sentences into fixed length and fixed number of small chunks
        batch_chunck_x, batch_chunck_y = DynamicChunkSplitTrainingData(batch_x, batch_y, m=62, C=11, n=1)
        return batch_chunck_x, batch_chunck_y
