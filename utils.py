#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: winston lin
"""
import pandas as pd
import numpy as np
from keras import backend as K


def getPaths(path_label, split_set):
    """
    This function is for filtering data by different constraints of label
    Args:
        path_label$ (str): path of label.
        split_set$ (str): 'Train', 'Validation' or 'Test' are supported.
        emo_attr$ (str): 'Act', 'Dom' or 'Val'
    """
    label_table = pd.read_csv(path_label)
    whole_fnames = (label_table['FileName'].values).astype('str')
    split_sets = (label_table['Split_Set'].values).astype('str')
    emo_act = label_table['EmoAct'].values
    emo_dom = label_table['EmoDom'].values
    emo_val = label_table['EmoVal'].values
    _paths = []
    _label_act = []
    _label_dom = []
    _label_val = []
    for i in range(len(whole_fnames)):
        # Constrain with Split Sets      
        if split_sets[i]==split_set:
            # Constrain with Emotional Labels
            _paths.append(whole_fnames[i])
            _label_act.append(emo_act[i])
            _label_dom.append(emo_dom[i])
            _label_val.append(emo_val[i])
        else:
            pass
    labels = []
    for a, d, v in zip(_label_act, _label_dom, _label_val):
        labels.append([a, d, v])
    return np.array(_paths), np.array(labels)

# Combining list of data arrays into a single large array
def CombineListToMatrix(Data):
    length_all = []
    for i in range(len(Data)):
        length_all.append(len(Data[i])) 
    feat_num = len(Data[0].T)
    Data_All = np.zeros((sum(length_all),feat_num))
    idx = 0
    Idx = []
    for i in range(len(length_all)):
        idx = idx+length_all[i]
        Idx.append(idx)        
    for i in range(len(Idx)):
        if i==0:    
            start = 0
            end = Idx[i]
            Data_All[start:end]=Data[i]
        else:
            start = Idx[i-1]
            end = Idx[i]
            Data_All[start:end]=Data[i]
    return Data_All      
    
# evaluated by CCC metric
def evaluation_metrics(true_value,predicted_value):
    corr_coeff_list = []
    ccc_list = []
    for i in range(true_value.shape[-1]):
        corr_coeff = np.corrcoef(true_value[:, i], predicted_value[:, i])
        ccc = 2*predicted_value[:, i].std()*true_value[:, i].std()*corr_coeff[0,1]/(predicted_value[:, i].var() + true_value[:, i].var() + (predicted_value[:, i].mean() - true_value[:, i].mean())**2)
        corr_coeff_list.append(corr_coeff)
        ccc_list.append(ccc)
    return (ccc_list, sum(ccc_list)/3)
    
    # corr_coeff = np.corrcoef(true_value,predicted_value)
    # ccc = 2*predicted_value.std()*true_value.std()*corr_coeff[0,1]/(predicted_value.var() + true_value.var() + (predicted_value.mean() - true_value.mean())**2)
    # return(ccc,corr_coeff)

# CCC loss function
def cc_coef(y_true, y_pred):
    mu_y_true_act, mu_y_true_dom, mu_y_true_val = K.mean(y_true[:, 0]), K.mean(y_true[:, 1]), K.mean(y_true[:, 2])
    mu_y_pred_act, mu_y_pred_dom, mu_y_pred_val = K.mean(y_pred[:, 0]), K.mean(y_pred[:, 1]), K.mean(y_pred[:, 2])
    loss_act = calculate_loss(y_true[:, 0], y_pred[:, 0], mu_y_true_act, mu_y_pred_act)
    loss_dom = calculate_loss(y_true[:, 1], y_pred[:, 1], mu_y_true_dom, mu_y_pred_dom)
    loss_val = calculate_loss(y_true[:, 2], y_pred[:, 2], mu_y_true_val, mu_y_pred_val)
    print(f"loss_act, loss_dom, loss_val: {loss_act}, {loss_dom}, {loss_val}")
    alpha, beta = (0.6, 0.2)
    return alpha * loss_act + beta * loss_dom + (1 - alpha - beta) * loss_val
                                                                                                                                   
def calculate_loss(y_true, y_pred, mu_y_true, mu_y_pred):
    return 1 - 2 * K.mean((y_true - mu_y_true) * (y_pred - mu_y_pred)) / (K.var(y_true) + K.var(y_pred) + K.mean(K.square(mu_y_pred - mu_y_true)))

# split original batch data into batch small-chunks data with
# proposed dynamic window step size which depends on the sentence duration 
def DynamicChunkSplitTrainingData(Batch_data, Batch_label, m, C, n):
    """
    Note! This function can't process sequence length which less than given m=62
    (e.g., 1sec=62frames, if LLDs extracted by hop size 16ms then 16ms*62=0.992sec~=1sec)
    Please make sure all your input data's length are greater then given m.
    
    Args:
         Batch_data$ (list): list of data arrays for a single batch.
        Batch_label$ (list): list of training targets for a single batch.
                  m$ (int) : chunk window length (i.e., number of frames within a chunk)
                  C$ (int) : number of chunks splitted for a sentence
                  n$ (int) : scaling factor to increase number of chunks splitted in a sentence
    """
    num_shifts = n*C-1  # Tmax = 11sec (for the MSP-Podcast corpus), 
                        # chunk needs to shift 10 times to obtain total C=11 chunks for each sentence
    Split_Data = []
    Split_Label = Split_Label_act = Split_Label_dom = Split_Label_val = np.array([])
    for i in range(len(Batch_data)):
        data = Batch_data[i]
        label = Batch_label[i]       
        # window-shifting size varied by differenct length of input utterance => dynamic step size
        step_size = int(int(len(data)-m)/num_shifts)      
        # Calculate index of chunks
        start_idx = [0]
        end_idx = [m]
        for iii in range(num_shifts):
            start_idx.extend([start_idx[0] + (iii+1)*step_size])
            end_idx.extend([end_idx[0] + (iii+1)*step_size])    
        # Output Split Data
        for iii in range(len(start_idx)):
            Split_Data.append( data[start_idx[iii]: end_idx[iii]] )    
        # Output Split Label
        split_label_act = np.repeat( label[0],len(start_idx) )
        split_label_dom = np.repeat( label[1],len(start_idx) )
        split_label_val = np.repeat( label[2],len(start_idx) )
        
        Split_Label_act = np.concatenate((Split_Label_act, split_label_act))
        Split_Label_dom = np.concatenate((Split_Label_dom, split_label_dom))
        Split_Label_val = np.concatenate((Split_Label_val, split_label_val))
    Split_Label = np.dstack((Split_Label_act, Split_Label_dom, Split_Label_val))
    Split_Label = Split_Label.reshape(Split_Label.shape[1], Split_Label.shape[2])

    # print(f"training_data_shape: {np.array(Split_Data).shape}")
    # print(f"training_label_shape: {Split_Label.shape}")
    return np.array(Split_Data), Split_Label

# split original batch data into batch small-chunks data with
# proposed dynamic window step size which depends on the sentence duration 
def DynamicChunkSplitTestingData(Online_data, m, C, n):
    """
    Note! This function can't process sequence length which less than given m=62
    (e.g., 1sec=62frames, if LLDs extracted by hop size 16ms then 16ms*62=0.992sec~=1sec)
    Please make sure all your input data's length are greater then given m.
    
    Args:
         Online_data$ (list): list of data array for a single sentence
                   m$ (int) : chunk window length (i.e., number of frames within a chunk)
                   C$ (int) : number of chunks splitted for a sentence
                   n$ (int) : scaling factor to increase number of chunks splitted in a sentence
    """
    num_shifts = n*C-1  # Tmax = 11sec (for the MSP-Podcast corpus), 
                        # chunk needs to shift 10 times to obtain total C=11 chunks for each sentence
    Split_Data = []
    for i in range(len(Online_data)):
        data = Online_data[i]
        # window-shifting size varied by differenct length of input utterance => dynamic step size
        step_size = int(int(len(data)-m)/num_shifts)      
        # Calculate index of chunks
        start_idx = [0]
        end_idx = [m]
        for iii in range(num_shifts):
            start_idx.extend([start_idx[0] + (iii+1)*step_size])
            end_idx.extend([end_idx[0] + (iii+1)*step_size])    
        # Output Split Data
        for iii in range(len(start_idx)):
            Split_Data.append( data[start_idx[iii]: end_idx[iii]] )
    return np.array(Split_Data)
