from __future__ import division, print_function
import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from ColonTissueLib import *
import warnings
warnings.filterwarnings("ignore")

# Ensure reproducible results.
random_state = 14

# Read data.
colon_full        = pd.read_csv('colonCancerData.csv', index_col=0)
colon_labels      = pd.read_csv('label.csv')
colon_labels      = ConvertLabels(colon_labels)
gene_descriptions = pd.read_csv('colonGeneDescriptions.csv')

# Split data into train dataset (85%) and test dataset (15%).
X_train_full, X_test, y_train_full, y_test = train_test_split(colon_full, colon_labels, test_size=0.15, random_state=random_state)

# Further split train dataset (85%) and validation dataset (15%).
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.15, random_state=random_state)

# Standardize data.
scaler     = StandardScaler().fit(X_train)
X_train_sc = scaler.transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)

# Establish baseline.
dummy_classifier_most_frequent = DummyClassifier(strategy='most_frequent')
dummy_classifier_most_frequent.fit(X_train_sc, y_train)

y_most_frequent          = dummy_classifier_most_frequent.predict(X_val_sc)
pred_proba_most_frequent = dummy_classifier_most_frequent.predict_proba(X_val_sc)

print("Baseline classification accuracy on validation set (Most frequent class): {:.3f}".
      format(accuracy_score(y_val, y_most_frequent)))    # 0.625
print("Baseline log-loss on validation set (Most frequent class): {}".
      format(log_loss(y_val, pred_proba_most_frequent)))    # 12.952041148091507

dummy_classifier_random = DummyClassifier(strategy='uniform', random_state=random_state)
dummy_classifier_random.fit(X_train_sc, y_train)

y_random          = dummy_classifier_random.predict(X_val_sc)
pred_proba_random = dummy_classifier_random.predict_proba(X_val_sc)

print("Baseline classification accuracy on validation set (Random prediction): {:.3f}".
      format(accuracy_score(y_val, y_random)))    # 0.750
print("Baseline log-loss on validation set (Random prediction): {}".
      format(log_loss(y_val, pred_proba_random)))    # 0.6931471805599453

# Select features.
# TODO: Consider PCA, kernel PCA, F-test, MFA, MFA-plus.
X_train_tnom_sc, X_train_tnom_descriptions = TNoM(X_train_sc, y_train, gene_descriptions, 20)







cv = KFold(n_splits=X_train_tnom_sc.shape[0], shuffle=True, random_state=random_state)
linear_svc_classifier = SVC(kernel='linear')

space  = [Real(10**-3, 10**3, "uniform", name="C"), # C
          Real(10**-4, 10**1, "uniform", name="gamma")] # gamma
x0 = [1, 10**-2]

def objective_linear_svc(params):    
    C, gamma = params
    linear_svc_classifier.set_params(C=C, gamma=gamma)
    
    return -np.mean(cross_val_score(linear_svc_classifier, X_train_tnom_sc, y_train, cv=cv, n_jobs=-1, scoring="accuracy"))

res_gp = gp_minimize(func=objective_linear_svc, dimensions=space, x0=x0, 
                     n_calls=25, random_state=random_state, n_random_starts=5, kappa=1.9)
print("Best score with Bayesian optimisation: {:.3f}".format(-res_gp.fun))
print("Best parameters with Bayesian optimisation:\nC: {}\ngamma: {}"
      .format(res_gp.x[0],res_gp.x[1]))











