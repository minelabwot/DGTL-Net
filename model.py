
# coding: utf-8

# In[4]:

import torch
import torch.nn as nn
import torch.nn.functional as F


# In[ ]:

class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(in_features=192, out_features=2, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # x = x.view(-1, 192)  # reshape tensor
        # x = self.fc(x)
        return x


# In[ ]:

class Generator(nn.Module):
    def __init__(self,):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        #self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        #self.pool3 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        #x = self.conv3(x)
        #x = F.relu(x)
       # x = self.pool3(x)

        return x


# In[ ]:

'''class Classifier_1(nn.Module):
    def __init__(self):
        super(Classifier_1, self).__init__()
        self.classifier_1 = nn.Sequential(
            nn.Linear(in_features=192, out_features=48),
            nn.ReLU(),
            nn.Linear(in_features=48, out_features=12),
            nn.ReLU(),
            nn.Linear(in_features=12, out_features=2)
        )
    def forward(self, x):

        x = self.classifier_1(x)

        return F.softmax(x)'''
class Classifier_1(nn.Module):
    def __init__(self):
        super(Classifier_1, self).__init__()
        self.fc1 = nn.Linear(in_features=192, out_features=48, bias=True)
        self.fc2 = nn.Linear(in_features=48, out_features=2, bias=True)
        #self.fc3 = nn.Linear(in_features=12, out_features=2, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        y = self.fc2(x)
        #x = F.relu(x)
        #y = self.fc3(x)
		
        return y


# In[ ]:

class Classifier_2(nn.Module):
    def __init__(self):
        super(Classifier_2, self).__init__()
        self.fc1 = nn.Linear(in_features=192, out_features=48, bias=True)
        self.fc2 = nn.Linear(in_features=48, out_features=2, bias=True)
        #self.fc3 = nn.Linear(in_features=12, out_features=2, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        y = self.fc2(x)
        #x = F.relu(x)
        #y = self.fc3(x)
        return y

