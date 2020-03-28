# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 19:59:35 2020

@author: Administrator
"""

def f_test(data, label, desc, k=10):
    '''
    perform f test to all the feature of input data and return
    the features with top scores. The order is in decreasing
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
    
    select = SelectKBest(k=k).fit(data, label)
    
    # Get rows of selected features
    score = select.scores_
    index = score.argsort()[-k:][::-1]
    X = data[:, index]
    Xd = desc.loc[index, :]
    
    return X,Xd

def TNoM(data, label, desc, k):
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
        for j in range(62):
            
            #perform desicion stump
            t = single_gene[j]
            temp = single_gene.copy()
            temp[single_gene <= t] = 0
            temp[single_gene > t] = 1
            
            #record the accuracy
            gene_acc.append(np.sum(temp.reshape(-1,1) != label.values))
            
            #change the direction and repeat
            temp = single_gene.copy()
            temp[single_gene <= t] = 1
            temp[single_gene > t] = 0
            
            gene_acc.append(np.sum(temp.reshape(-1,1) != label.values))
            
        TNoM.append(np.min(gene_acc))
    
    #get index in increase order of TNoM score
    index = sorted(range(len(TNoM)), key=lambda i: TNoM[i])[0:k]
    
    #get the rows of feature
    X = data[:, index]
    Xd = desc.loc[index, :]
    
    return X,Xd

#usage template
#train, train_dec = TNoM(colon_scale, colon_label, gene, 20)