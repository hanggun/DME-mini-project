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

#-------------------------------------------------------------------
# MFA has three functions
#-------------------------------------------------------------------
def MFA(data, label, desc, k):
    '''
    perform MFA Score to all the features of input data and return
    the features with top MFA scores. The order is in increasing
    order of scores. The default number of nearest neiborhoods is 5
    and can be change manully.
    
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
    
    #5 nearest neighbors
    k1 = KNN1(data, label, 5)
    k2 = KNN2(data, label, 5)
    
    #calculate Ww
    nrow = data.shape[0]
    ncol = data.shape[1]
    
    W_w = np.ones([nrow,nrow])
    for i in range(nrow):
        for j in range(nrow):
            if (i in k1[j]) or (j in k1[i]):
                W_w[i,j] = 1
            else:
                W_w[i,j] = 0
    
    #calculate Wb
    W_b = np.ones([nrow,nrow])
    for i in range(nrow):
        for j in range(nrow):
            if (i in k2[j]) or (j in k2[i]):
                W_b[i,j] = 1
            else:
                W_b[i,j] = 0
                
    D_w = np.diag(np.sum(W_w, axis = 0))
    D_b = np.diag(np.sum(W_b, axis = 0))
    
    L_w = D_w - W_w
    L_b = D_b - W_b
    
    MFA_score = np.ones(ncol)
    for i in range(ncol):
        MFA_score[i] = (data[:,i].reshape(-1,1).T @ L_b @ data[:,i].reshape(-1,1)) /\
        (data[:,i].reshape(-1,1).T @ L_w @ data[:,i].reshape(-1,1))
      
    index = MFA_score.argsort()[-k:][::-1]
    X = data[:, index]
    Xd = desc.loc[index, :]
    
    return X,Xd

def KNN1(data, label, k):
    '''
    Calculate k nearest neiborhoods of sample x of same labels and record the index
    '''

    k1 = np.ones([62,k])
    index0 = label.index[label['label'] == 0].tolist()
    index1 = label.index[label['label'] == 1].tolist()
    
    for i in range(62):
        dist = []
        for j in range(62):
            if(i != j):
                #calculate euclidean distance
                distance = np.sqrt(np.sum((data[i,:] - data[j,:])**2))
                dist.append(distance)
            else:
                dist.append(100000)
        
        if i in index0:
            a = pd.DataFrame(dist)
            b = a.loc[index0]
            c = b.sort_values(0)
            k1[i, :] = list(c[0:k].index)
        else:
            a = pd.DataFrame(dist)
            b = a.loc[index1]
            c = b.sort_values(0)
            k1[i, :] = list(c[0:k].index)
            
    return k1

def KNN2(data, label, k):
    '''
    Calculate k nearest neiborhoods of sample x of different labels and record the index
    '''

    k2 = np.ones([62,k])
    index0 = label.index[label['label'] == 0].tolist()
    index1 = label.index[label['label'] == 1].tolist()
    
    for i in range(62):
        dist = []
        for j in range(62):
            if(i != j):
                #calculate euclidean distance
                distance = np.sqrt(np.sum((data[i,:] - data[j,:])**2))
                dist.append(distance)
            else:
                dist.append(100000)
        
        if i in index0:
            a = pd.DataFrame(dist)
            b = a.loc[index1]
            c = b.sort_values(0)
            k2[i, :] = list(c[0:k].index)
        else:
            a = pd.DataFrame(dist)
            b = a.loc[index0]
            c = b.sort_values(0)
            k2[i, :] = list(c[0:k].index)
            
    return k2

#usage template
#train, train_dec = TNoM(colon_scale, colon_label, gene, 20)
#train, train_dec = MFA(colon_scale, colon_label, gene, 10)