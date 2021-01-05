
# coding: utf-8

# In[ ]:

import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import MMD
import numpy as np
import random
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import os
import csv
random.seed(5)


# In[ ]:

def Net_A(data_S_H,data_S_F,label_S,extractor,generator,classifier_1,epochs,LR,i):
    print("Net_A training")
    inputs_normal = torch.tensor(data_S_H.values).float()
    inputs_normal = inputs_normal.unsqueeze(1)
    inputs_fault = torch.tensor(data_S_F.values).float()
    inputs_fault = inputs_fault.unsqueeze(1)
    file = 'loss_record/loss_record_A.csv'
    round_info = pd.DataFrame({'round_info':[["round:",i]]})
    round_info.to_csv(file,index=False,mode='a+',header=False)
    for epoch in range(epochs):
        print('Round:{},Epoch: {}'.format(i,epoch))
        
        extractor = extractor.train()
        generator = generator.train()
        classifier_1 = classifier_1.train()
        
        
        classifier_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(list(extractor.parameters())+list(generator.parameters())+list(classifier_1.parameters())
                                    , lr=LR, betas=(0.9, 0.999))
        optimizer.zero_grad()
       
        normal_S_feature = extractor(inputs_normal) 
        fault_S_feature = extractor(inputs_fault)
        S_generator = generator(normal_S_feature)
        generator_loss = MMD.mmd_rbf_noaccelerate(S_generator,fault_S_feature)
        feature_total = torch.cat([normal_S_feature, fault_S_feature])
        feature_total = feature_total.view(feature_total.shape[0], -1)
        feature_total = torch.tensor(feature_total)
        label_total = torch.tensor(label_S.values)
        label_total = label_total.squeeze_()
        result_classifier = classifier_1(feature_total) 
        classifier_loss = classifier_criterion(result_classifier, label_total)
        loss_A = generator_loss+classifier_loss
        loss_A.backward()
        optimizer.step()
        _, predicted = torch.max(result_classifier.data, 1)
        print("generator_loss",generator_loss)
        print("classifier_loss",classifier_loss)
        print(metrics.classification_report(label_total.data,predicted))
        print("confusion_matrix:",confusion_matrix(label_total.data, predicted, labels=None, sample_weight=None))
        print("Loss_A:",loss_A)   
        loss_data = pd.DataFrame({'Net_A_generator_loss':[generator_loss.item()],'Net_A_classifier_loss':[classifier_loss.item()],'Loss_A':[loss_A.item()]})
        loss_data.to_csv(file,index=False,mode='a+',header=False)

    torch.save(extractor.state_dict(),"extractor.pkl")
    torch.save(generator.state_dict(),"generator.pkl")
    torch.save(classifier_1.state_dict(),"classifier_1.pkl")

        


# In[ ]:

def Net_B(data_S_H,data_S_F,data_T_H,label_S,extractor,generator,classifier_2,epochs,LR,i):
    print("Net_B training")
    data_S_normal = torch.tensor(data_S_H.values).float()
    data_S_normal = data_S_normal.unsqueeze(1)
    data_S_fault = torch.tensor(data_S_F.values).float()
    data_S_fault = data_S_fault.unsqueeze(1)
    data_T_normal = torch.tensor(data_T_H.values).float()
    data_T_normal = data_T_normal.unsqueeze(1)
    file = 'loss_record/loss_record_B.csv'
    round_info = pd.DataFrame({'round_info':[["round:",i]]})
    round_info.to_csv(file,index=False,mode='a+',header=False)
    for epoch in range(epochs):
        print('Round:{},Epoch: {}'.format(i,epoch))
        extractor = extractor.train()
        classifier_2 = classifier_2.train()
        
        #classifier_criterion =  nn.CrossEntropyLoss()
        classifier_criterion =  nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([0.3,1.0])).float(),size_average=True)
        optimizer = torch.optim.Adam(list(extractor.parameters())+list(classifier_2.parameters())
                                    , lr=LR, betas=(0.9, 0.999))
        optimizer.zero_grad()
        
        normal_S_feature = extractor(data_S_normal) 
        fault_S_feature = extractor(data_S_fault)
        normal_T_feature = extractor(data_T_normal)
        
        T_generator = generator(normal_T_feature)
        
        total_S_feature = torch.cat([normal_S_feature, fault_S_feature], dim=0)
        total_T_feature = torch.cat([normal_T_feature, T_generator], dim=0)
        
        transfer_loss = MMD.mmd_rbf_noaccelerate(total_S_feature,total_T_feature)
        
        feature_total = torch.cat([total_S_feature, total_T_feature])
        label_total = pd.concat([label_S,label_S])
        feature_total = feature_total.view(feature_total.shape[0], -1)
        feature_total = torch.Tensor(feature_total)
        label_total = torch.tensor(label_total.values)
        label_total = label_total.squeeze_()
        inputData = feature_total
        result = classifier_2(inputData)
        classifier_loss = classifier_criterion(result, label_total)
        loss_B = transfer_loss+classifier_loss
        loss_B.backward()
        optimizer.step()
        _, predicted = torch.max(result.data, 1)
        print("transfer_loss",transfer_loss)
        print("classifier_loss",classifier_loss)
        print(metrics.classification_report(label_total.data,predicted))
        print("confusion_matrix:",confusion_matrix(label_total.data, predicted, labels=None, sample_weight=None))
        print("Loss_B:",loss_B)     
        loss_data = pd.DataFrame({'Net_B_transfer_loss':[transfer_loss.item()],'Net_B_classifier_loss':[classifier_loss.item()],'Loss_B':[loss_B.item()]})
        loss_data.to_csv(file,index=False,mode='a+',header=False)

    torch.save(extractor.state_dict(),"extractor.pkl")
    torch.save(classifier_2.state_dict(),"classifier_2.pkl")


