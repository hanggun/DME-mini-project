from __future__ import division, print_function
import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score


def ConvertLabels(labels, convertToString=False):
    '''
    Convert labels to +1 or -1. +1 means normal tissue, -1 means tumor tissue.
    '''
    column_name = 'label'
    label       = labels.copy()
    
    if convertToString == False:
        label.loc[labels[column_name] > 0]  = 1
        label.loc[labels[column_name] <= 0] = -1
        
    else:
        label.loc[labels[column_name] > 0]  = 'normal'
        label.loc[labels[column_name] <= 0] = 'tumor'

    return label

def TNoM(data, label, desc, k=10):
    '''
    perform TNoM Score to all the features of input data and return
    the features with top TNoM scores. The order is in increasing
    order of scores
    
    Input:
        -------------------------------------------
        data:
            scaled data
        label:
            data label
        desc:
            data description
        k:
            number of feature return(default is 10)
            
    return:
        -------------------------------------------
        X:
            feature selected
        Xd:
            data description
    '''  
    TNoM = []

    for i in range(2000):
        #get single gene
        single_gene = data[:, i]
        gene_acc = []
        for j in range(data.shape[0]):
            
            #perform desicion stump
            t = single_gene[j] + 1e-6
            temp = single_gene.copy()
            temp[single_gene < t] = -1
            temp[single_gene > t] = 1
            
            #record the accuracy
            gene_acc.append(np.sum(temp.reshape(-1,1) != label.values))
            
            #change the direction and repeat
            temp = single_gene.copy()
            temp[single_gene < t] = 1
            temp[single_gene > t] = -1
            
            gene_acc.append(np.sum(temp.reshape(-1,1) != label.values))
            
            #perform desicion stump
            t = single_gene[j] - 1e-6
            temp = single_gene.copy()
            temp[single_gene < t] = -1
            temp[single_gene > t] = 1
            
            #record the accuracy
            gene_acc.append(np.sum(temp.reshape(-1,1) != label.values))
            
            #change the direction and repeat
            temp = single_gene.copy()
            temp[single_gene < t] = 1
            temp[single_gene > t] = -1
            
            gene_acc.append(np.sum(temp.reshape(-1,1) != label.values))
            
        TNoM.append(np.min(gene_acc))
    
    #get index in increase order of TNoM score
    index = sorted(range(len(TNoM)), key=lambda i: TNoM[i])[0:k]
    
    #get the rows of feature
    X = data[:, index]
    TNoM = pd.DataFrame({'TNoM':TNoM})
    Xd = desc.loc[index, :]
    Xd = Xd.join(TNoM.loc[index,:])
    
    return X,Xd































