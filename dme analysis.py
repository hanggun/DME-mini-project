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
        label.loc[labels[column_name] <= 0] = 0
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
pca = PCA(n_components=2)
X_pca_2d = pca.fit_transform(colon_scale.T)
scatter_2d_label(X_pca_2d, gene_desc)

#--------------------------------------------------------
#perform T-SNE feature selection
#--------------------------------------------------------
#set perplexity
#perl = [2, 5, 10, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
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
X_new = SelectKBest(k=20).fit_transform(X, Y)

