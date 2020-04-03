# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 13:31:31 2020

@author: Administrator
"""
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

from sklearn.utils import (as_float_array, check_array, check_X_y, safe_sqr,
                     safe_mask)
from scipy import special, stats

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

#Load data 
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


#Plot heatmap
pd_scale = pd.DataFrame(colon_scale)
pd_scale.columns = gene['1']
corr = np.round(pd_scale.iloc[:, 0:10].corr(method='pearson'),4)
fig, ax = plt.subplots(1, 1, figsize = (10,6))
fig = sns.heatmap(np.abs(corr), annot=True, fmt=".1f", linewidths=.5)
plt.title('The Heat Map of Correlation between each Feature')
plt.xlabel('Gene Index')
plt.ylabel('Gene Index')
plt.savefig('Gene heatmap', dpi=100)