
# coding: utf-8

# In[2]:

import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import model
import train


# In[ ]:

if __name__ == "__main__":
    data_S_H = pd.read_csv('data/normal_B_sample.csv')
    data_S_F = pd.read_csv('data/fault_B_sample.csv')
    data_T_H = pd.read_csv('data/normal_A_sample.csv')
    label_S = pd.read_csv('data/label_B.csv')
    epochs_A = 100
    epochs_B = 100
    extractor = model.Extractor()
    generator = model.Generator()
    
    classifier_1 = model.Classifier_1()
    classifier_2 = model.Classifier_2()
	
    LR_A = 0.00001
    LR_B = 0.00001
    extractor.load_state_dict(torch.load('1D_CNN.pkl')) 
    for i in range(50):
        print('All Net',i)
        
        train.Net_A(data_S_H,data_S_F,label_S,extractor,generator
                    ,classifier_1,epochs_A,LR_A,i)
        extractor.load_state_dict(torch.load('extractor.pkl'))
        generator.load_state_dict(torch.load('generator.pkl'))
        train.Net_B(data_S_H,data_S_F,data_T_H,label_S,extractor
                    ,generator,classifier_2,epochs_B,LR_B,i)
        extractor.load_state_dict(torch.load('extractor.pkl'))
        
        
        
        
        
    

