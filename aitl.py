import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import sklearn.preprocessing as sk
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from torch.autograd import Function


#######################################################
#             AITL Classes & Functions                #          
#######################################################

class FX(nn.Module):
    def __init__(self, dropout_rate, input_dim, h_dim, zdim, act='relu'):
        super(FX, self).__init__()
        if act == 'relu':
            act = nn.ReLU()
        else:
            act = nn.LeakyReLU()
        self.EnE = torch.nn.Sequential(
            nn.Linear(input_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            act,
            nn.Dropout(p=dropout_rate))
    def forward(self, x):
        output = self.EnE(x)
        return output

class MTL(nn.Module):
    def __init__(self, dropout_rate, h_dim, z_dim, act='relu'):
        super(MTL, self).__init__()
        self.act = act
        if act == 'relu':
            self.actf = nn.ReLU()
        else:
            self.actf = nn.LeakyReLU()
        self.Sh = nn.Linear(h_dim, z_dim)
        self.bn1 = nn.BatchNorm1d(z_dim)
        self.Drop = nn.Dropout(p=dropout_rate)
        self.Source = torch.nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.Dropout(p=dropout_rate),
            self.actf,
            nn.Linear(z_dim, 1))
        self.Target = torch.nn.Sequential(
            nn.Linear(z_dim, 1),
            nn.Sigmoid())        
    def forward(self, S, T):
        if S is None:
            if self.act == 'relu':
                ZT = F.relu(self.Drop(self.bn1(self.Sh((T)))))
            else:
                ZT = F.leaky_relu(self.Drop(self.bn1(self.Sh((T)))))
            yhat_S = None
            yhat_T = self.Target(ZT)
        elif T is None:
            if self.act == 'relu':
                ZS = F.relu(self.Drop(self.bn1(self.Sh((S)))))
            else:
                ZS = F.leaky_relu(self.Drop(self.bn1(self.Sh((S)))))
            yhat_S = self.Source(ZS)
            yhat_T = None
        else:
            if self.act == 'relu'
                ZS = F.relu(self.Drop(self.bn1(self.Sh((S)))))
                ZT = F.relu(self.Drop(self.bn1(self.Sh((T)))))
            else:
                ZS = F.leaky_relu(self.Drop(self.bn1(self.Sh((S)))))
                ZT = F.leaky_relu(self.Drop(self.bn1(self.Sh((T)))))
            yhat_S = self.Source(ZS)
            yhat_T = self.Target(ZT)
        return yhat_S, yhat_T   

class GradReverse(Function):
    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -1)

def grad_reverse(x):
    return GradReverse()(x)

class Discriminator(nn.Module):
    def __init__(self, dropout_rate, h_dim, z_dim):
        super(Discriminator, self).__init__()
        self.D1 = nn.Linear(h_dim, 1)
        self.Drop1 = nn.Dropout(p=dropout_rate)
        self.Drop2 = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = grad_reverse(x)
        yhat = self.Drop1(self.D1(x))
        return torch.sigmoid(yhat)


## Early stopper class

class EarlyStopper:
    def __init__(self, patience=1, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.min_valloss = np.inf

    def __call__(self, valloss):
        if valloss < self.min_valloss:
            self.min_valloss = valloss
            self.counter = 0
        elif valloss > (self.min_valloss + self.delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
