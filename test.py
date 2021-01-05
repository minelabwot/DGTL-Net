
# coding: utf-8

# In[1]:

import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import model
import train
from sklearn import metrics
from sklearn.metrics import confusion_matrix


# In[ ]:

if __name__ == "__main__":
    extractor = model.Extractor()
    classifier_2 = model.Classifier_2()
    extractor.load_state_dict(torch.load('extractor.pkl'))
    classifier_2.load_state_dict(torch.load('classifier_2.pkl'))
    
    test_data = pd.read_csv('test_data/data4.csv')
    test_label = pd.read_csv('test_data/label4.csv')
    inputs_data = torch.tensor(test_data.values).float()
    inputs_data = inputs_data.unsqueeze(1)
    inputs_label = torch.tensor(test_label.values)
    inputs_label = inputs_label.squeeze_()
    data_feature = extractor(inputs_data) 
    data_feature = data_feature.view(data_feature.shape[0], -1)
    data_feature = torch.Tensor(data_feature)
    result = classifier_2(data_feature)
    _, predicted = torch.max(result.data, 1)
    print(metrics.classification_report(inputs_label, predicted)) 
    print("confusion_matrix:")
    print(confusion_matrix(inputs_label, predicted, labels=None, sample_weight=None))
    
   

