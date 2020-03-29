from __future__ import division, print_function # Imports from __future__ since we're running Python 2
import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import log_loss
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import GridSearchCV,KFold,LeaveOneOut
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier

from sklearn.feature_selection import SelectKBest

from sklearn.manifold import Isomap
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
import umap

from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage 

#--------------------------------------------------------
#functions
#--------------------------------------------------------
def ConvertLabels(labels, convert=False):
    '''
    This function is used to convert the label to 0 and 1 range.
    Label 1 means normal tissue, label 0 means tumor tissue
    '''
    
    column_name = 'label'
    #Convert to 0 and 1
    label = labels.copy()
    if convert == False:
        label.loc[labels[column_name] > 0] = 1
        label.loc[labels[column_name] <= 0] = -1
    else:
        label.loc[labels[column_name] > 0] = 'normal'
        label.loc[labels[column_name] <= 0] = 'tumor'
    
    #Convert 1 to normal and 0 to tumor
    
    
    return label

def scatter_2d_label(X_2d, y, s=2, alpha=0.5, lw=2):
    """Visualuse a 2D embedding with corresponding labels.
    
    X_2d : ndarray, shape (n_samples,2)
        Low-dimensional feature representation.
    
    y : ndarray, shape (n_samples,)
        Labels corresponding to the entries in X_2d.
        
    s : float
        Marker size for scatter plot.
    
    alpha : float
        Transparency for scatter plot.
        
    lw : float
        Linewidth for scatter plot.
    """
    targets = pd.unique(y)
    colors = sns.color_palette(n_colors=targets.size)
    for color, target in zip(colors, targets):
        plt.scatter(X_2d[y == target, 0], X_2d[y == target, 1],
                    color=color, label=target, s=s, alpha=alpha, lw=lw)
#--------------------------------------------------------
#preprocessing
#--------------------------------------------------------
# Download colon data and label
colon = pd.read_csv('colonCancerData.csv', index_col=0) # Gene expression
colon_label = pd.read_csv('label.csv') # labels 
gene = pd.read_csv('colonGeneDescriptions.csv')
gene_desc = gene['2']
gene_desc.loc[pd.isna(gene_desc)] = 'NA'

#Convert the colon label
colon_label = ConvertLabels(colon_label)

#standardize data
scaler = StandardScaler()
colon_scale = scaler.fit_transform(colon)

#--------------------------------------------------------
#perform PCA on gene
#--------------------------------------------------------
def pca():
    '''
    Plot the 2D figure of PCA with the first two PCs on gene
    wise direction
    '''
    
    pca = PCA(n_components=2)
    X_pca_2d = pca.fit_transform(colon_scale.T)
    scatter_2d_label(X_pca_2d, gene_desc)

#--------------------------------------------------------
#perform T-SNE feature selection
#--------------------------------------------------------
#set perplexity
#perl = [2, 5, 10, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
def t_sne():
    '''
    Plot the figure after t_sne feature selection on gene-wise direction
    '''
    
    perl = [35]
    sns.set(font_scale=1.5) # Set default font size
    fig, ax = plt.subplots(1,1,figsize=(12,10))
    for ii, perplexity in enumerate(perl):
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=10)
        colon_tsne_2d = tsne.fit_transform(colon_scale.T)
        plt.subplot(1,1,ii+1)
        scatter_2d_label(colon_tsne_2d, gene_desc)
        plt.title('Perplexity: {}, KL-score: {}'.format(perplexity, tsne.kl_divergence_))
        plt.xlabel('Component 1')
        plt.ylabel('Component 2 ')
    plt.legend(loc='center left', bbox_to_anchor=[1.01, 1], scatterpoints=3)
    fig.tight_layout()
    plt.show()


#--------------------------------------------------------
#Baseline Classifier
#--------------------------------------------------------
# most_frequent strategy
X = colon_scale
Y = colon_label

clf = DummyClassifier(strategy='most_frequent')

cv = KFold(X.shape[0], True, random_state=1234)
cv_results = cross_validate(clf, X, np.c_[Y].reshape(-1,), cv=cv)

ca_score = np.mean(cv_results['test_score'])

clf_name = 'Dummy Classifer'
print ("{}, accuracy: {:.3f}".format(clf_name, ca_score))

#--------------------------------------------------------
#univariate test with F-value
#--------------------------------------------------------
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
        for j in range(62):
            
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


def KNN1(data, label, k):
    '''
    Calculate k nearest neiborhoods of sample x of same labels and record the index
    '''

    k1 = np.ones([62,k])
    index0 = label.index[label['label'] == -1].tolist()
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
    
#calculate k nearest neiborhoods of sample x of different labels and record the index   
def KNN2(data, label, k=5):
    '''
    Calculate k nearest neiborhoods of sample x of different labels and record the index
    '''

    k2 = np.ones([62,k])
    index0 = label.index[label['label'] == -1].tolist()
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
        
def MFA(data, label, desc, k_neighbor=5, k=10):
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
    k1 = KNN1(data, label, k_neighbor)
    k2 = KNN2(data, label, k_neighbor)
    
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

from scipy.stats import pearsonr
def MFAplus(data, label, desc, threshold, k_neighbor=5, k=10):
    '''
    perform MFA+ Score to all the features of input data and return
    the features with top MFA scores. The order is in increasing
    order of scores.
    
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
    feature, feature_desc = MFA(colon_scale, colon_label, desc, k_neighbor, k=2000)
    X = feature[:,0].reshape(-1,1)
    Xd = pd.DataFrame()
    Xd = pd.concat([Xd,feature_desc.iloc[[0], :]])
    
    j = 0
    while X.shape[1] < k:
        j += 1
        flag = 0
        for i in range(X.shape[1]):
            corr, p = pearsonr(X[:,i], feature[:,j])
            
            if corr > threshold:
                flag = 1
        
        if flag == 0:
            X = np.append(X,feature[:,j].reshape(-1,1),axis=1)
            Xd = pd.concat([Xd,feature_desc.iloc[[j], :]])
        
    return X,Xd

#train, train_dec = f_test(colon_scale, colon_label, gene, 20)
train, train_dec = TNoM(colon_scale, colon_label, gene, 20)
#train, train_dec = MFA(colon_scale, colon_label, gene)
#train, train_dec = MFAplus(colon_scale, colon_label, gene, 0.7)