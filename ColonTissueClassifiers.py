from __future__ import division, print_function
import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from skopt import gp_minimize
from skopt.plots import plot_convergence
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import pickle
from ColonTissueLib import *
import warnings
warnings.filterwarnings("ignore")

# Ensure reproducible results.
random_state = 12

# Read data.
colon_full        = pd.read_csv('colonCancerData.csv', index_col=0)
colon_labels      = pd.read_csv('label.csv')
colon_labels      = ConvertLabels(colon_labels)
gene_descriptions = pd.read_csv('colonGeneDescriptions.csv')

# Split data into train dataset (80%) and test dataset (20%).
X_train, X_test, y_train, y_test = train_test_split(colon_full, colon_labels, test_size=0.2, random_state=random_state)

# Standardize data.
scaler     = StandardScaler().fit(X_train)
X_train_sc = scaler.transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Define scorers.
precision_scorer = make_scorer(precision_score)
recall_scorer    = make_scorer(recall_score)
f1_scorer        = make_scorer(f1_score)
accuracy_scorer  = make_scorer(accuracy_score)
log_loss_scorer  = make_scorer(log_loss, labels=[1, -1])

# Establish baseline.
dummy_classifier_most_frequent = DummyClassifier(strategy='most_frequent')

cv         = KFold(n_splits=X_train_sc.shape[0], shuffle=True, random_state=random_state)
scoring    = {'accuracy': accuracy_scorer, 'log_loss': log_loss_scorer}
cv_results = cross_validate(dummy_classifier_most_frequent, X_train_sc, y_train, cv=cv, scoring=scoring)

dummy_classifier_most_frequent_cv_accuracy  = np.mean(cv_results['test_accuracy'])
dummy_classifier_most_frequent_cv_log_loss  = np.mean(cv_results['test_log_loss'])

print("Baseline LOOCV accuracy (Most frequent class): {:.4f}".format(dummy_classifier_most_frequent_cv_accuracy))    # 0.653
print("Baseline LOOCV log-loss (Most frequent class): {}".format(dummy_classifier_most_frequent_cv_log_loss))    # 11.982840790071053

dummy_classifier_random = DummyClassifier(strategy='uniform', random_state=random_state)

cv         = KFold(n_splits=X_train_sc.shape[0], shuffle=True, random_state=random_state)
scoring    = {'accuracy': accuracy_scorer, 'log_loss': log_loss_scorer}
cv_results = cross_validate(dummy_classifier_random, X_train_sc, y_train, cv=cv, scoring=scoring)

dummy_classifier_random_cv_accuracy  = np.mean(cv_results['test_accuracy'])
dummy_classifier_random_cv_log_loss  = np.mean(cv_results['test_log_loss'])

print("Baseline LOOCV accuracy (Random prediction): {:.4f}".format(dummy_classifier_random_cv_accuracy))    # 0.347
print("Baseline LOOCV log-loss (Random prediction): {}".format(dummy_classifier_random_cv_log_loss))    # 22.556457790916486

# Set feature counts to be considered.
feature_counts = [i + 1 for i in range(50)]

'''
Linear SVM Classifier
'''

# Linear SVM classifier with F-test.
linear_svm_f_test_index  = []
linear_svm_f_test_scores = []
linear_svm_f_test_Cs     = []
linear_svm_f_test_gammas = []

for feature_count in feature_counts:
    print("Fine tune linear SVM classifier (F-test) with k = {}".format(feature_count))
    
    X_train_f_test_sc, X_train_f_test_descriptions, f_test_index = f_test(X_train_sc, y_train, gene_descriptions, feature_count)

    cv = KFold(n_splits=X_train_f_test_sc.shape[0], shuffle=True, random_state=random_state)
    
    linear_svm_classifier = SVC(kernel='linear')
    
    space = [Real(10**-3, 10**3, "uniform", name="C"),
             Real(10**-4, 10**1, "uniform", name="gamma")]
    x0    = [1, 10**-2]
    
    @use_named_args(space)
    def objective_linear_svm(**params):    
        linear_svm_classifier.set_params(**params)
        return -np.mean(cross_val_score(linear_svm_classifier, X_train_f_test_sc, y_train, cv=cv, n_jobs=-1, scoring="accuracy"))
    
    res_gp = gp_minimize(func=objective_linear_svm, dimensions=space, x0=x0, n_calls=50, random_state=random_state, n_random_starts=5, kappa=1.9)
    
    print("Best score = {}".format(-res_gp.fun))
    
    linear_svm_f_test_index.append(f_test_index)
    linear_svm_f_test_scores.append(-res_gp.fun)
    linear_svm_f_test_Cs.append(res_gp.x[0])
    linear_svm_f_test_gammas.append(res_gp.x[1])

best_linear_svm_f_test_index = linear_svm_f_test_scores.index(max(linear_svm_f_test_scores))

linear_svm_f_test_dict = {"linear_svm_f_test_index": linear_svm_f_test_index,
                          "linear_svm_f_test_scores": linear_svm_f_test_scores,
                          "linear_svm_f_test_Cs": linear_svm_f_test_Cs,
                          "linear_svm_f_test_gammas": linear_svm_f_test_gammas,
                          "best_linear_svm_f_test_index": best_linear_svm_f_test_index}
outputPickleFileHandle = open("linear_svm_f_test_dict.pkl", "wb")
pickle.dump(linear_svm_f_test_dict, outputPickleFileHandle)
outputPickleFileHandle.close()

# Linear SVM classifier with TNoM.
linear_svm_tnom_index  = []
linear_svm_tnom_scores = []
linear_svm_tnom_Cs     = []
linear_svm_tnom_gammas = []

for feature_count in feature_counts:
    print("Fine tune linear SVM classifier (TNoM) with k = {}".format(feature_count))
    
    X_train_tnom_sc, X_train_tnom_descriptions, tnom_index = TNoM(X_train_sc, y_train, gene_descriptions, feature_count)

    cv = KFold(n_splits=X_train_tnom_sc.shape[0], shuffle=True, random_state=random_state)
    
    linear_svm_classifier = SVC(kernel='linear')
    
    space = [Real(10**-3, 10**3, "uniform", name="C"),
             Real(10**-4, 10**1, "uniform", name="gamma")]
    x0    = [1, 10**-2]
    
    @use_named_args(space)
    def objective_linear_svm(**params):    
        linear_svm_classifier.set_params(**params)
        return -np.mean(cross_val_score(linear_svm_classifier, X_train_tnom_sc, y_train, cv=cv, n_jobs=-1, scoring="accuracy"))
    
    res_gp = gp_minimize(func=objective_linear_svm, dimensions=space, x0=x0, n_calls=50, random_state=random_state, n_random_starts=5, kappa=1.9)
    
    print("Best score = {}".format(-res_gp.fun))
    
    linear_svm_tnom_index.append(tnom_index)
    linear_svm_tnom_scores.append(-res_gp.fun)
    linear_svm_tnom_Cs.append(res_gp.x[0])
    linear_svm_tnom_gammas.append(res_gp.x[1])

best_linear_svm_tnom_index = linear_svm_tnom_scores.index(max(linear_svm_tnom_scores))

linear_svm_tnom_dict = {"linear_svm_tnom_index": linear_svm_tnom_index,
                        "linear_svm_tnom_scores": linear_svm_tnom_scores,
                        "linear_svm_tnom_Cs": linear_svm_tnom_Cs,
                        "linear_svm_tnom_gammas": linear_svm_tnom_gammas,
                        "best_linear_svm_tnom_index": best_linear_svm_tnom_index}
outputPickleFileHandle = open("linear_svm_tnom_dict.pkl", "wb")
pickle.dump(linear_svm_tnom_dict, outputPickleFileHandle)
outputPickleFileHandle.close()

# Linear SVM classifier with MFA.
linear_svm_mfa_index  = []
linear_svm_mfa_scores = []
linear_svm_mfa_Cs     = []
linear_svm_mfa_gammas = []

for feature_count in feature_counts:
    print("Fine tune linear SVM classifier (MFA) with k = {}".format(feature_count))
    
    X_train_mfa_sc, X_train_mfa_descriptions, mfa_index = MFA(X_train_sc, y_train, gene_descriptions, k_neighbor=5, k=feature_count)

    cv = KFold(n_splits=X_train_mfa_sc.shape[0], shuffle=True, random_state=random_state)
    
    linear_svm_classifier = SVC(kernel='linear')
    
    space = [Real(10**-3, 10**3, "uniform", name="C"),
             Real(10**-4, 10**1, "uniform", name="gamma")]
    x0    = [1, 10**-2]
    
    @use_named_args(space)
    def objective_linear_svm(**params):    
        linear_svm_classifier.set_params(**params)
        return -np.mean(cross_val_score(linear_svm_classifier, X_train_mfa_sc, y_train, cv=cv, n_jobs=-1, scoring="accuracy"))
    
    res_gp = gp_minimize(func=objective_linear_svm, dimensions=space, x0=x0, n_calls=50, random_state=random_state, n_random_starts=5, kappa=1.9)
    
    print("Best score = {}".format(-res_gp.fun))
    
    linear_svm_mfa_index.append(mfa_index)
    linear_svm_mfa_scores.append(-res_gp.fun)
    linear_svm_mfa_Cs.append(res_gp.x[0])
    linear_svm_mfa_gammas.append(res_gp.x[1])

best_linear_svm_mfa_index = linear_svm_mfa_scores.index(max(linear_svm_mfa_scores))

linear_svm_mfa_dict = {"linear_svm_mfa_index": linear_svm_mfa_index,
                       "linear_svm_mfa_scores": linear_svm_mfa_scores,
                       "linear_svm_mfa_Cs": linear_svm_mfa_Cs,
                       "linear_svm_mfa_gammas": linear_svm_mfa_gammas,
                       "best_linear_svm_mfa_index": best_linear_svm_mfa_index}
outputPickleFileHandle = open("linear_svm_mfa_dict.pkl", "wb")
pickle.dump(linear_svm_mfa_dict, outputPickleFileHandle)
outputPickleFileHandle.close()

# Linear SVM classifier with MFA-plus.
linear_svm_mfa_plus_index  = []
linear_svm_mfa_plus_scores = []
linear_svm_mfa_plus_Cs     = []
linear_svm_mfa_plus_gammas = []

for feature_count in feature_counts:
    print("Fine tune linear SVM classifier (MFA-plus) with k = {}".format(feature_count))
    
    X_train_mfa_plus_sc, X_train_mfa_plus_descriptions, mfa_plus_index = MFAplus(X_train_sc, y_train, gene_descriptions, threshold=0.9, k_neighbor=5, k=feature_count)

    cv = KFold(n_splits=X_train_mfa_plus_sc.shape[0], shuffle=True, random_state=random_state)
    
    linear_svm_classifier = SVC(kernel='linear')
    
    space = [Real(10**-3, 10**3, "uniform", name="C"),
             Real(10**-4, 10**1, "uniform", name="gamma")]
    x0    = [1, 10**-2]
    
    @use_named_args(space)
    def objective_linear_svm(**params):    
        linear_svm_classifier.set_params(**params)
        return -np.mean(cross_val_score(linear_svm_classifier, X_train_mfa_plus_sc, y_train, cv=cv, n_jobs=-1, scoring="accuracy"))
    
    res_gp = gp_minimize(func=objective_linear_svm, dimensions=space, x0=x0, n_calls=50, random_state=random_state, n_random_starts=5, kappa=1.9)
    
    print("Best score = {}".format(-res_gp.fun))
    
    linear_svm_mfa_plus_index.append(mfa_plus_index)
    linear_svm_mfa_plus_scores.append(-res_gp.fun)
    linear_svm_mfa_plus_Cs.append(res_gp.x[0])
    linear_svm_mfa_plus_gammas.append(res_gp.x[1])

best_linear_svm_mfa_plus_index = linear_svm_mfa_plus_scores.index(max(linear_svm_mfa_plus_scores))

linear_svm_mfa_plus_dict = {"linear_svm_mfa_plus_index": linear_svm_mfa_plus_index,
                            "linear_svm_mfa_plus_scores": linear_svm_mfa_plus_scores,
                            "linear_svm_mfa_plus_Cs": linear_svm_mfa_plus_Cs,
                            "linear_svm_mfa_plus_gammas": linear_svm_mfa_plus_gammas,
                            "best_linear_svm_mfa_plus_index": best_linear_svm_mfa_plus_index}
outputPickleFileHandle = open("linear_svm_mfa_plus_dict.pkl", "wb")
pickle.dump(linear_svm_mfa_plus_dict, outputPickleFileHandle)
outputPickleFileHandle.close()

'''
Gaussian Naive Bayes Classifier
'''

# Gaussian Naive Bayes classifier with F-test.
gaussian_nb_f_test_index          = []
gaussian_nb_f_test_scores         = []
gaussian_nb_f_test_var_smoothings = []

for feature_count in feature_counts:
    print("Fine tune Gaussian Naive Bayes classifier (F-test) with k = {}".format(feature_count))

    X_train_f_test_sc, X_train_f_test_descriptions, f_test_index = f_test(X_train_sc, y_train, gene_descriptions, feature_count)

    cv = KFold(n_splits=X_train_f_test_sc.shape[0], shuffle=True, random_state=random_state)
    
    gaussian_nb_classifier = GaussianNB()
    
    space = [Real(10**-12, 10**-6, "uniform", name="var_smoothing")]
    
    @use_named_args(space)
    def objective_gaussian_nb(**params):
        gaussian_nb_classifier.set_params(**params)
        return -np.mean(cross_val_score(gaussian_nb_classifier, X_train_f_test_sc, y_train, cv=cv, n_jobs=-1, scoring="accuracy"))

    res_gp = gp_minimize(func=objective_gaussian_nb, dimensions=space, n_calls=50, random_state=random_state, n_random_starts=5, kappa=1.9)

    print("Best score = {}".format(-res_gp.fun))

    gaussian_nb_f_test_index.append(f_test_index)
    gaussian_nb_f_test_scores.append(-res_gp.fun)
    gaussian_nb_f_test_var_smoothings.append(res_gp.x[0])

best_gaussian_nb_f_test_index = gaussian_nb_f_test_scores.index(max(gaussian_nb_f_test_scores))

gaussian_nb_f_test_dict = {"gaussian_nb_f_test_index": gaussian_nb_f_test_index,
                           "gaussian_nb_f_test_scores": gaussian_nb_f_test_scores,
                           "gaussian_nb_f_test_var_smoothings": gaussian_nb_f_test_var_smoothings,
                           "best_gaussian_nb_f_test_index": best_gaussian_nb_f_test_index}
outputPickleFileHandle = open("gaussian_nb_f_test_dict.pkl", "wb")
pickle.dump(gaussian_nb_f_test_dict, outputPickleFileHandle)
outputPickleFileHandle.close()

# Gaussian Naive Bayes classifier with TNoM.
gaussian_nb_tnom_index          = []
gaussian_nb_tnom_scores         = []
gaussian_nb_tnom_var_smoothings = []

for feature_count in feature_counts:
    print("Fine tune Gaussian Naive Bayes classifier (TNoM) with k = {}".format(feature_count))

    X_train_tnom_sc, X_train_tnom_descriptions, tnom_index = TNoM(X_train_sc, y_train, gene_descriptions, feature_count)

    cv = KFold(n_splits=X_train_tnom_sc.shape[0], shuffle=True, random_state=random_state)
    
    gaussian_nb_classifier = GaussianNB()
    
    space = [Real(10**-12, 10**-6, "uniform", name="var_smoothing")]
    
    @use_named_args(space)
    def objective_gaussian_nb(**params):
        gaussian_nb_classifier.set_params(**params)
        return -np.mean(cross_val_score(gaussian_nb_classifier, X_train_tnom_sc, y_train, cv=cv, n_jobs=-1, scoring="accuracy"))

    res_gp = gp_minimize(func=objective_gaussian_nb, dimensions=space, n_calls=50, random_state=random_state, n_random_starts=5, kappa=1.9)

    print("Best score = {}".format(-res_gp.fun))

    gaussian_nb_tnom_index.append(tnom_index)
    gaussian_nb_tnom_scores.append(-res_gp.fun)
    gaussian_nb_tnom_var_smoothings.append(res_gp.x[0])

best_gaussian_nb_tnom_index = gaussian_nb_tnom_scores.index(max(gaussian_nb_tnom_scores))

gaussian_nb_tnom_dict = {"gaussian_nb_tnom_index": gaussian_nb_tnom_index,
                         "gaussian_nb_tnom_scores": gaussian_nb_tnom_scores,
                         "gaussian_nb_tnom_var_smoothings": gaussian_nb_tnom_var_smoothings,
                         "best_gaussian_nb_tnom_index": best_gaussian_nb_tnom_index}
outputPickleFileHandle = open("gaussian_nb_tnom_dict.pkl", "wb")
pickle.dump(gaussian_nb_tnom_dict, outputPickleFileHandle)
outputPickleFileHandle.close()

# Gaussian Naive Bayes classifier with MFA.
gaussian_nb_mfa_index          = []
gaussian_nb_mfa_scores         = []
gaussian_nb_mfa_var_smoothings = []

for feature_count in feature_counts:
    print("Fine tune Gaussian Naive Bayes classifier (MFA) with k = {}".format(feature_count))

    X_train_mfa_sc, X_train_mfa_descriptions, mfa_index = MFA(X_train_sc, y_train, gene_descriptions, k_neighbor=5, k=feature_count)

    cv = KFold(n_splits=X_train_mfa_sc.shape[0], shuffle=True, random_state=random_state)
    
    gaussian_nb_classifier = GaussianNB()
    
    space = [Real(10**-12, 10**-6, "uniform", name="var_smoothing")]
    
    @use_named_args(space)
    def objective_gaussian_nb(**params):
        gaussian_nb_classifier.set_params(**params)
        return -np.mean(cross_val_score(gaussian_nb_classifier, X_train_mfa_sc, y_train, cv=cv, n_jobs=-1, scoring="accuracy"))

    res_gp = gp_minimize(func=objective_gaussian_nb, dimensions=space, n_calls=50, random_state=random_state, n_random_starts=5, kappa=1.9)

    print("Best score = {}".format(-res_gp.fun))

    gaussian_nb_mfa_index.append(mfa_index)
    gaussian_nb_mfa_scores.append(-res_gp.fun)
    gaussian_nb_mfa_var_smoothings.append(res_gp.x[0])

best_gaussian_nb_mfa_index = gaussian_nb_mfa_scores.index(max(gaussian_nb_mfa_scores))

gaussian_nb_mfa_dict = {"gaussian_nb_mfa_index": gaussian_nb_mfa_index,
                        "gaussian_nb_mfa_scores": gaussian_nb_mfa_scores,
                        "gaussian_nb_mfa_var_smoothings": gaussian_nb_mfa_var_smoothings,
                        "best_gaussian_nb_mfa_index": best_gaussian_nb_mfa_index}
outputPickleFileHandle = open("gaussian_nb_mfa_dict.pkl", "wb")
pickle.dump(gaussian_nb_mfa_dict, outputPickleFileHandle)
outputPickleFileHandle.close()

# Gaussian Naive Bayes classifier with MFA-plus.
gaussian_nb_mfa_plus_index          = []
gaussian_nb_mfa_plus_scores         = []
gaussian_nb_mfa_plus_var_smoothings = []

for feature_count in feature_counts:
    print("Fine tune Gaussian Naive Bayes classifier (MFA-plus) with k = {}".format(feature_count))

    X_train_mfa_plus_sc, X_train_mfa_plus_descriptions, mfa_plus_index = MFAplus(X_train_sc, y_train, gene_descriptions, threshold=0.9, k_neighbor=5, k=feature_count)

    cv = KFold(n_splits=X_train_mfa_plus_sc.shape[0], shuffle=True, random_state=random_state)
    
    gaussian_nb_classifier = GaussianNB()
    
    space = [Real(10**-12, 10**-6, "uniform", name="var_smoothing")]
    
    @use_named_args(space)
    def objective_gaussian_nb(**params):
        gaussian_nb_classifier.set_params(**params)
        return -np.mean(cross_val_score(gaussian_nb_classifier, X_train_mfa_plus_sc, y_train, cv=cv, n_jobs=-1, scoring="accuracy"))

    res_gp = gp_minimize(func=objective_gaussian_nb, dimensions=space, n_calls=50, random_state=random_state, n_random_starts=5, kappa=1.9)

    print("Best score = {}".format(-res_gp.fun))

    gaussian_nb_mfa_plus_index.append(mfa_plus_index)
    gaussian_nb_mfa_plus_scores.append(-res_gp.fun)
    gaussian_nb_mfa_plus_var_smoothings.append(res_gp.x[0])

best_gaussian_nb_mfa_plus_index = gaussian_nb_mfa_plus_scores.index(max(gaussian_nb_mfa_plus_scores))

gaussian_nb_mfa_plus_dict = {"gaussian_nb_mfa_plus_index": gaussian_nb_mfa_plus_index,
                             "gaussian_nb_mfa_plus_scores": gaussian_nb_mfa_plus_scores,
                             "gaussian_nb_mfa_plus_var_smoothings": gaussian_nb_mfa_plus_var_smoothings,
                             "best_gaussian_nb_mfa_plus_index": best_gaussian_nb_mfa_plus_index}
outputPickleFileHandle = open("gaussian_nb_mfa_plus_dict.pkl", "wb")
pickle.dump(gaussian_nb_mfa_plus_dict, outputPickleFileHandle)
outputPickleFileHandle.close()

'''
KNN Classifier
'''
# KNN classifier with F-test.
knn_f_test_index       = []
knn_f_test_scores      = []
knn_f_test_n_neighbors = []
knn_f_test_leaf_sizes  = []
knn_f_test_ps          = []

for feature_count in feature_counts:
    print("Fine tune KNN classifier (F-test) with k = {}".format(feature_count))

    X_train_f_test_sc, X_train_f_test_descriptions, f_test_index = f_test(X_train_sc, y_train, gene_descriptions, feature_count)

    cv = KFold(n_splits=X_train_f_test_sc.shape[0], shuffle=True, random_state=random_state)

    knn_classifier = KNeighborsClassifier()

    space = [Integer(1, 20, name="n_neighbors"),
             Integer(5, 55, name="leaf_size"),
             Integer(1, 10, name="p")]

    @use_named_args(space)
    def objective_knn(**params):
        knn_classifier.set_params(**params)
        return -np.mean(cross_val_score(knn_classifier, X_train_f_test_sc, y_train, cv=cv, n_jobs=-1, scoring="accuracy"))
    
    res_gp = gp_minimize(func=objective_knn, dimensions=space, n_calls=50, random_state=random_state, n_random_starts=5, kappa=1.9)
    
    print("Best score = {}".format(-res_gp.fun))
    
    knn_f_test_index.append(f_test_index)
    knn_f_test_scores.append(-res_gp.fun)
    knn_f_test_n_neighbors.append(res_gp.x[0])
    knn_f_test_leaf_sizes.append(res_gp.x[1])
    knn_f_test_ps.append(res_gp.x[2])

best_knn_f_test_index = knn_f_test_scores.index(max(knn_f_test_scores))

knn_f_test_dict = {"knn_f_test_index": knn_f_test_index,
                   "knn_f_test_scores": knn_f_test_scores,
                   "knn_f_test_n_neighbors": knn_f_test_n_neighbors,
                   "knn_f_test_leaf_sizes": knn_f_test_leaf_sizes,
                   "knn_f_test_ps": knn_f_test_ps,
                   "best_knn_f_test_index": best_knn_f_test_index}
outputPickleFileHandle = open("knn_f_test_dict.pkl", "wb")
pickle.dump(knn_f_test_dict, outputPickleFileHandle)
outputPickleFileHandle.close()

# KNN classifier with TNoM.
knn_tnom_index       = []
knn_tnom_scores      = []
knn_tnom_n_neighbors = []
knn_tnom_leaf_sizes  = []
knn_tnom_ps          = []

for feature_count in feature_counts:
    print("Fine tune KNN classifier (TNoM) with k = {}".format(feature_count))

    X_train_tnom_sc, X_train_tnom_descriptions, tnom_index = TNoM(X_train_sc, y_train, gene_descriptions, feature_count)

    cv = KFold(n_splits=X_train_tnom_sc.shape[0], shuffle=True, random_state=random_state)

    knn_classifier = KNeighborsClassifier()

    space = [Integer(1, 20, name="n_neighbors"),
             Integer(5, 55, name="leaf_size"),
             Integer(1, 10, name="p")]

    @use_named_args(space)
    def objective_knn(**params):
        knn_classifier.set_params(**params)
        return -np.mean(cross_val_score(knn_classifier, X_train_tnom_sc, y_train, cv=cv, n_jobs=-1, scoring="accuracy"))
    
    res_gp = gp_minimize(func=objective_knn, dimensions=space, n_calls=50, random_state=random_state, n_random_starts=5, kappa=1.9)
    
    print("Best score = {}".format(-res_gp.fun))
    
    knn_tnom_index.append(tnom_index)
    knn_tnom_scores.append(-res_gp.fun)
    knn_tnom_n_neighbors.append(res_gp.x[0])
    knn_tnom_leaf_sizes.append(res_gp.x[1])
    knn_tnom_ps.append(res_gp.x[2])

best_knn_tnom_index = knn_tnom_scores.index(max(knn_tnom_scores))

knn_tnom_dict = {"knn_tnom_index": knn_tnom_index,
                 "knn_tnom_scores": knn_tnom_scores,
                 "knn_tnom_n_neighbors": knn_tnom_n_neighbors,
                 "knn_tnom_leaf_sizes": knn_tnom_leaf_sizes,
                 "knn_tnom_ps": knn_tnom_ps,
                 "best_knn_tnom_index": best_knn_tnom_index}
outputPickleFileHandle = open("knn_tnom_dict.pkl", "wb")
pickle.dump(knn_tnom_dict, outputPickleFileHandle)
outputPickleFileHandle.close()

# KNN classifier with MFA.
knn_mfa_index       = []
knn_mfa_scores      = []
knn_mfa_n_neighbors = []
knn_mfa_leaf_sizes  = []
knn_mfa_ps          = []

for feature_count in feature_counts:
    print("Fine tune KNN classifier (MFA) with k = {}".format(feature_count))

    X_train_mfa_sc, X_train_mfa_descriptions, mfa_index = MFA(X_train_sc, y_train, gene_descriptions, k_neighbor=5, k=feature_count)

    cv = KFold(n_splits=X_train_mfa_sc.shape[0], shuffle=True, random_state=random_state)

    knn_classifier = KNeighborsClassifier()

    space = [Integer(1, 20, name="n_neighbors"),
             Integer(5, 55, name="leaf_size"),
             Integer(1, 10, name="p")]

    @use_named_args(space)
    def objective_knn(**params):
        knn_classifier.set_params(**params)
        return -np.mean(cross_val_score(knn_classifier, X_train_mfa_sc, y_train, cv=cv, n_jobs=-1, scoring="accuracy"))
    
    res_gp = gp_minimize(func=objective_knn, dimensions=space, n_calls=50, random_state=random_state, n_random_starts=5, kappa=1.9)
    
    print("Best score = {}".format(-res_gp.fun))
    
    knn_mfa_index.append(mfa_index)
    knn_mfa_scores.append(-res_gp.fun)
    knn_mfa_n_neighbors.append(res_gp.x[0])
    knn_mfa_leaf_sizes.append(res_gp.x[1])
    knn_mfa_ps.append(res_gp.x[2])

best_knn_mfa_index = knn_mfa_scores.index(max(knn_mfa_scores))

knn_mfa_dict = {"knn_mfa_index": knn_mfa_index,
                "knn_mfa_scores": knn_mfa_scores,
                "knn_mfa_n_neighbors": knn_mfa_n_neighbors,
                "knn_mfa_leaf_sizes": knn_mfa_leaf_sizes,
                "knn_mfa_ps": knn_mfa_ps,
                "best_knn_mfa_index": best_knn_mfa_index}
outputPickleFileHandle = open("knn_mfa_dict.pkl", "wb")
pickle.dump(knn_mfa_dict, outputPickleFileHandle)
outputPickleFileHandle.close()

# KNN classifier with MFA-plus.
knn_mfa_plus_index       = []
knn_mfa_plus_scores      = []
knn_mfa_plus_n_neighbors = []
knn_mfa_plus_leaf_sizes  = []
knn_mfa_plus_ps          = []

for feature_count in feature_counts:
    print("Fine tune KNN classifier (MFA-plus) with k = {}".format(feature_count))

    X_train_mfa_plus_sc, X_train_mfa_plus_descriptions, mfa_plus_index = MFAplus(X_train_sc, y_train, gene_descriptions, threshold=0.9, k_neighbor=5, k=feature_count)

    cv = KFold(n_splits=X_train_mfa_plus_sc.shape[0], shuffle=True, random_state=random_state)

    knn_classifier = KNeighborsClassifier()

    space = [Integer(1, 20, name="n_neighbors"),
             Integer(5, 55, name="leaf_size"),
             Integer(1, 10, name="p")]

    @use_named_args(space)
    def objective_knn(**params):
        knn_classifier.set_params(**params)
        return -np.mean(cross_val_score(knn_classifier, X_train_mfa_plus_sc, y_train, cv=cv, n_jobs=-1, scoring="accuracy"))
    
    res_gp = gp_minimize(func=objective_knn, dimensions=space, n_calls=50, random_state=random_state, n_random_starts=5, kappa=1.9)
    
    print("Best score = {}".format(-res_gp.fun))
    
    knn_mfa_plus_index.append(mfa_plus_index)
    knn_mfa_plus_scores.append(-res_gp.fun)
    knn_mfa_plus_n_neighbors.append(res_gp.x[0])
    knn_mfa_plus_leaf_sizes.append(res_gp.x[1])
    knn_mfa_plus_ps.append(res_gp.x[2])

best_knn_mfa_plus_index = knn_mfa_plus_scores.index(max(knn_mfa_plus_scores))

knn_mfa_plus_dict = {"knn_mfa_plus_index": knn_mfa_plus_index,
                     "knn_mfa_plus_scores": knn_mfa_plus_scores,
                     "knn_mfa_plus_n_neighbors": knn_mfa_plus_n_neighbors,
                     "knn_mfa_plus_leaf_sizes": knn_mfa_plus_leaf_sizes,
                     "knn_mfa_plus_ps": knn_mfa_plus_ps,
                     "best_knn_mfa_plus_index": best_knn_mfa_plus_index}
outputPickleFileHandle = open("knn_mfa_plus_dict.pkl", "wb")
pickle.dump(knn_mfa_plus_dict, outputPickleFileHandle)
outputPickleFileHandle.close()

'''
Logistic Regression Classifier
'''
# Logistic regression classifier with F-test.
logistic_regression_f_test_index  = []
logistic_regression_f_test_scores = []
logistic_regression_f_test_Cs     = []

for feature_count in feature_counts:
    print("Fine tune logistic regression classifier (F-test) with k = {}".format(feature_count))
    
    X_train_f_test_sc, X_train_f_test_descriptions, f_test_index = f_test(X_train_sc, y_train, gene_descriptions, feature_count)

    cv = KFold(n_splits=X_train_f_test_sc.shape[0], shuffle=True, random_state=random_state)
    
    logistic_regression_classifier = LogisticRegression()
    
    space = [Real(1.0, 10**3, "uniform", name="C")]
    
    @use_named_args(space)
    def objective_logistic_regression(**params):
        logistic_regression_classifier.set_params(**params)
        return -np.mean(cross_val_score(logistic_regression_classifier, X_train_f_test_sc, y_train, cv=cv, n_jobs=-1, scoring="accuracy"))
    
    res_gp = gp_minimize(func=objective_logistic_regression, dimensions=space, n_calls=50, random_state=random_state, n_random_starts=5, kappa=1.9)
    
    print("Best score = {}".format(-res_gp.fun))
    
    logistic_regression_f_test_index.append(f_test_index)
    logistic_regression_f_test_scores.append(-res_gp.fun)
    logistic_regression_f_test_Cs.append(res_gp.x[0])

best_logistic_regression_f_test_index = logistic_regression_f_test_scores.index(max(logistic_regression_f_test_scores))

logistic_regression_f_test_dict = {"logistic_regression_f_test_index": logistic_regression_f_test_index,
                                   "logistic_regression_f_test_scores": logistic_regression_f_test_scores,
                                   "logistic_regression_f_test_Cs": logistic_regression_f_test_Cs,
                                   "best_logistic_regression_f_test_index": best_logistic_regression_f_test_index}
outputPickleFileHandle = open("logistic_regression_f_test_dict.pkl", "wb")
pickle.dump(logistic_regression_f_test_dict, outputPickleFileHandle)
outputPickleFileHandle.close()

# Logistic regression classifier with TNoM.
logistic_regression_tnom_index  = []
logistic_regression_tnom_scores = []
logistic_regression_tnom_Cs     = []

for feature_count in feature_counts:
    print("Fine tune logistic regression classifier (TNoM) with k = {}".format(feature_count))
    
    X_train_tnom_sc, X_train_tnom_descriptions, tnom_index = TNoM(X_train_sc, y_train, gene_descriptions, feature_count)

    cv = KFold(n_splits=X_train_tnom_sc.shape[0], shuffle=True, random_state=random_state)
    
    logistic_regression_classifier = LogisticRegression()
    
    space = [Real(1.0, 10**3, "uniform", name="C")]
    
    @use_named_args(space)
    def objective_logistic_regression(**params):
        logistic_regression_classifier.set_params(**params)
        return -np.mean(cross_val_score(logistic_regression_classifier, X_train_tnom_sc, y_train, cv=cv, n_jobs=-1, scoring="accuracy"))
    
    res_gp = gp_minimize(func=objective_logistic_regression, dimensions=space, n_calls=50, random_state=random_state, n_random_starts=5, kappa=1.9)
    
    print("Best score = {}".format(-res_gp.fun))
    
    logistic_regression_tnom_index.append(tnom_index)
    logistic_regression_tnom_scores.append(-res_gp.fun)
    logistic_regression_tnom_Cs.append(res_gp.x[0])

best_logistic_regression_tnom_index = logistic_regression_tnom_scores.index(max(logistic_regression_tnom_scores))

logistic_regression_tnom_dict = {"logistic_regression_tnom_index": logistic_regression_tnom_index,
                                 "logistic_regression_tnom_scores": logistic_regression_tnom_scores,
                                 "logistic_regression_tnom_Cs": logistic_regression_tnom_Cs,
                                 "best_logistic_regression_tnom_index": best_logistic_regression_tnom_index}
outputPickleFileHandle = open("logistic_regression_tnom_dict.pkl", "wb")
pickle.dump(logistic_regression_tnom_dict, outputPickleFileHandle)
outputPickleFileHandle.close()

# Logistic regression classifier with MFA.
logistic_regression_mfa_index  = []
logistic_regression_mfa_scores = []
logistic_regression_mfa_Cs     = []

for feature_count in feature_counts:
    print("Fine tune logistic regression classifier (MFA) with k = {}".format(feature_count))
    
    X_train_mfa_sc, X_train_mfa_descriptions, mfa_index = MFA(X_train_sc, y_train, gene_descriptions, k_neighbor=5, k=feature_count)

    cv = KFold(n_splits=X_train_mfa_sc.shape[0], shuffle=True, random_state=random_state)
    
    logistic_regression_classifier = LogisticRegression()
    
    space = [Real(1.0, 10**3, "uniform", name="C")]
    
    @use_named_args(space)
    def objective_logistic_regression(**params):
        logistic_regression_classifier.set_params(**params)
        return -np.mean(cross_val_score(logistic_regression_classifier, X_train_mfa_sc, y_train, cv=cv, n_jobs=-1, scoring="accuracy"))
    
    res_gp = gp_minimize(func=objective_logistic_regression, dimensions=space, n_calls=50, random_state=random_state, n_random_starts=5, kappa=1.9)
    
    print("Best score = {}".format(-res_gp.fun))
    
    logistic_regression_mfa_index.append(mfa_index)
    logistic_regression_mfa_scores.append(-res_gp.fun)
    logistic_regression_mfa_Cs.append(res_gp.x[0])

best_logistic_regression_mfa_index = logistic_regression_mfa_scores.index(max(logistic_regression_mfa_scores))

logistic_regression_mfa_dict = {"logistic_regression_mfa_index": logistic_regression_mfa_index,
                                "logistic_regression_mfa_scores": logistic_regression_mfa_scores,
                                "logistic_regression_mfa_Cs": logistic_regression_mfa_Cs,
                                "best_logistic_regression_mfa_index": best_logistic_regression_mfa_index}
outputPickleFileHandle = open("logistic_regression_mfa_dict.pkl", "wb")
pickle.dump(logistic_regression_mfa_dict, outputPickleFileHandle)
outputPickleFileHandle.close()

# Logistic regression classifier with MFA-plus.
logistic_regression_mfa_plus_index  = []
logistic_regression_mfa_plus_scores = []
logistic_regression_mfa_plus_Cs     = []

for feature_count in feature_counts:
    print("Fine tune logistic regression classifier (MFA-plus) with k = {}".format(feature_count))
    
    X_train_mfa_plus_sc, X_train_mfa_plus_descriptions, mfa_plus_index = MFAplus(X_train_sc, y_train, gene_descriptions, threshold=0.9, k_neighbor=5, k=feature_count)

    cv = KFold(n_splits=X_train_mfa_plus_sc.shape[0], shuffle=True, random_state=random_state)
    
    logistic_regression_classifier = LogisticRegression()
    
    space = [Real(1.0, 10**3, "uniform", name="C")]
    
    @use_named_args(space)
    def objective_logistic_regression(**params):
        logistic_regression_classifier.set_params(**params)
        return -np.mean(cross_val_score(logistic_regression_classifier, X_train_mfa_plus_sc, y_train, cv=cv, n_jobs=-1, scoring="accuracy"))
    
    res_gp = gp_minimize(func=objective_logistic_regression, dimensions=space, n_calls=50, random_state=random_state, n_random_starts=5, kappa=1.9)
    
    print("Best score = {}".format(-res_gp.fun))
    
    logistic_regression_mfa_plus_index.append(mfa_plus_index)
    logistic_regression_mfa_plus_scores.append(-res_gp.fun)
    logistic_regression_mfa_plus_Cs.append(res_gp.x[0])

best_logistic_regression_mfa_plus_index = logistic_regression_mfa_plus_scores.index(max(logistic_regression_mfa_plus_scores))

logistic_regression_mfa_plus_dict = {"logistic_regression_mfa_plus_index": logistic_regression_mfa_plus_index,
                                     "logistic_regression_mfa_plus_scores": logistic_regression_mfa_plus_scores,
                                     "logistic_regression_mfa_plus_Cs": logistic_regression_mfa_plus_Cs,
                                     "best_logistic_regression_mfa_plus_index": best_logistic_regression_mfa_plus_index}
outputPickleFileHandle = open("logistic_regression_mfa_plus_dict.pkl", "wb")
pickle.dump(logistic_regression_mfa_plus_dict, outputPickleFileHandle)
outputPickleFileHandle.close()

'''
Multi-Layer Perceptron Classifier
'''
# Multi-layer perceptron classifier with F-test.
mlp_f_test_index              = []
mlp_f_test_scores             = []
mlp_f_test_hidden_layer_sizes = []
mlp_f_test_alphas             = []

for feature_count in feature_counts:
    print("Fine tune multi-layer perceptron classifier (F-test) with k = {}".format(feature_count))
    
    X_train_f_test_sc, X_train_f_test_descriptions, f_test_index = f_test(X_train_sc, y_train, gene_descriptions, feature_count)

    cv = KFold(n_splits=X_train_f_test_sc.shape[0], shuffle=True, random_state=random_state)
    
    mlp_classifier = MLPClassifier(random_state=random_state)
    
    space = [Integer(10, 1000, name="hidden_layer_sizes"),
             Real(10**-8, 1, name="alpha")]
    
    @use_named_args(space)
    def objective_mlp(**params):    
        mlp_classifier.set_params(**params)
        return -np.mean(cross_val_score(mlp_classifier, X_train_f_test_sc, y_train, cv=cv, n_jobs=-1, scoring="accuracy"))
    
    res_gp = gp_minimize(func=objective_mlp, dimensions=space, n_calls=50, random_state=random_state, n_random_starts=5, kappa=1.9)

    print("Best score = {}".format(-res_gp.fun))
    
    mlp_f_test_index.append(f_test_index)
    mlp_f_test_scores.append(-res_gp.fun)
    mlp_f_test_hidden_layer_sizes.append(res_gp.x[0])
    mlp_f_test_alphas.append(res_gp.x[1])

best_mlp_f_test_index = mlp_f_test_scores.index(max(mlp_f_test_scores))

mlp_f_test_dict = {"mlp_f_test_index": mlp_f_test_index,
                   "mlp_f_test_scores": mlp_f_test_scores,
                   "mlp_f_test_hidden_layer_sizes": mlp_f_test_hidden_layer_sizes,
                   "mlp_f_test_alphas": mlp_f_test_alphas,
                   "best_mlp_f_test_index": best_mlp_f_test_index}
outputPickleFileHandle = open("mlp_f_test_dict.pkl", "wb")
pickle.dump(mlp_f_test_dict, outputPickleFileHandle)
outputPickleFileHandle.close()

# Multi-layer perceptron classifier with TNoM.
mlp_tnom_index              = []
mlp_tnom_scores             = []
mlp_tnom_hidden_layer_sizes = []
mlp_tnom_alphas             = []

for feature_count in feature_counts:
    print("Fine tune multi-layer perceptron classifier (TNoM) with k = {}".format(feature_count))
    
    X_train_tnom_sc, X_train_tnom_descriptions, tnom_index = TNoM(X_train_sc, y_train, gene_descriptions, feature_count)

    cv = KFold(n_splits=X_train_tnom_sc.shape[0], shuffle=True, random_state=random_state)
    
    mlp_classifier = MLPClassifier(random_state=random_state)
    
    space = [Integer(10, 1000, name="hidden_layer_sizes"),
             Real(10**-8, 1, name="alpha")]
    
    @use_named_args(space)
    def objective_mlp(**params):    
        mlp_classifier.set_params(**params)
        return -np.mean(cross_val_score(mlp_classifier, X_train_tnom_sc, y_train, cv=cv, n_jobs=-1, scoring="accuracy"))
    
    res_gp = gp_minimize(func=objective_mlp, dimensions=space, n_calls=50, random_state=random_state, n_random_starts=5, kappa=1.9)

    print("Best score = {}".format(-res_gp.fun))
    
    mlp_tnom_index.append(tnom_index)
    mlp_tnom_scores.append(-res_gp.fun)
    mlp_tnom_hidden_layer_sizes.append(res_gp.x[0])
    mlp_tnom_alphas.append(res_gp.x[1])

best_mlp_tnom_index = mlp_tnom_scores.index(max(mlp_tnom_scores))

mlp_tnom_dict = {"mlp_tnom_index": mlp_tnom_index,
                 "mlp_tnom_scores": mlp_tnom_scores,
                 "mlp_tnom_hidden_layer_sizes": mlp_tnom_hidden_layer_sizes,
                 "mlp_tnom_alphas": mlp_tnom_alphas,
                 "best_mlp_tnom_index": best_mlp_tnom_index}
outputPickleFileHandle = open("mlp_tnom_dict.pkl", "wb")
pickle.dump(mlp_tnom_dict, outputPickleFileHandle)
outputPickleFileHandle.close()

# Multi-layer perceptron classifier with MFA.
mlp_mfa_index              = []
mlp_mfa_scores             = []
mlp_mfa_hidden_layer_sizes = []
mlp_mfa_alphas             = []

for feature_count in feature_counts:
    print("Fine tune multi-layer perceptron classifier (MFA) with k = {}".format(feature_count))
    
    X_train_mfa_sc, X_train_mfa_descriptions, mfa_index = MFA(X_train_sc, y_train, gene_descriptions, k_neighbor=5, k=feature_count)

    cv = KFold(n_splits=X_train_mfa_sc.shape[0], shuffle=True, random_state=random_state)
    
    mlp_classifier = MLPClassifier(random_state=random_state)
    
    space = [Integer(10, 1000, name="hidden_layer_sizes"),
             Real(10**-8, 1, name="alpha")]
    
    @use_named_args(space)
    def objective_mlp(**params):    
        mlp_classifier.set_params(**params)
        return -np.mean(cross_val_score(mlp_classifier, X_train_mfa_sc, y_train, cv=cv, n_jobs=-1, scoring="accuracy"))
    
    res_gp = gp_minimize(func=objective_mlp, dimensions=space, n_calls=50, random_state=random_state, n_random_starts=5, kappa=1.9)

    print("Best score = {}".format(-res_gp.fun))
    
    mlp_mfa_index.append(mfa_index)
    mlp_mfa_scores.append(-res_gp.fun)
    mlp_mfa_hidden_layer_sizes.append(res_gp.x[0])
    mlp_mfa_alphas.append(res_gp.x[1])

best_mlp_mfa_index = mlp_mfa_scores.index(max(mlp_mfa_scores))

mlp_mfa_dict = {"mlp_mfa_index": mlp_mfa_index,
                "mlp_mfa_scores": mlp_mfa_scores,
                "mlp_mfa_hidden_layer_sizes": mlp_mfa_hidden_layer_sizes,
                "mlp_mfa_alphas": mlp_mfa_alphas,
                "best_mlp_mfa_index": best_mlp_mfa_index}
outputPickleFileHandle = open("mlp_mfa_dict.pkl", "wb")
pickle.dump(mlp_mfa_dict, outputPickleFileHandle)
outputPickleFileHandle.close()

# Multi-layer perceptron classifier with MFA-plus.
mlp_mfa_plus_index              = []
mlp_mfa_plus_scores             = []
mlp_mfa_plus_hidden_layer_sizes = []
mlp_mfa_plus_alphas             = []

for feature_count in feature_counts:
    print("Fine tune multi-layer perceptron classifier (MFA-plus) with k = {}".format(feature_count))
    
    X_train_mfa_plus_sc, X_train_mfa_plus_descriptions, mfa_plus_index = MFAplus(X_train_sc, y_train, gene_descriptions, threshold=0.9, k_neighbor=5, k=feature_count)

    cv = KFold(n_splits=X_train_mfa_plus_sc.shape[0], shuffle=True, random_state=random_state)
    
    mlp_classifier = MLPClassifier(random_state=random_state)
    
    space = [Integer(10, 1000, name="hidden_layer_sizes"),
             Real(10**-8, 1, name="alpha")]
    
    @use_named_args(space)
    def objective_mlp(**params):    
        mlp_classifier.set_params(**params)
        return -np.mean(cross_val_score(mlp_classifier, X_train_mfa_plus_sc, y_train, cv=cv, n_jobs=-1, scoring="accuracy"))
    
    res_gp = gp_minimize(func=objective_mlp, dimensions=space, n_calls=50, random_state=random_state, n_random_starts=5, kappa=1.9)

    print("Best score = {}".format(-res_gp.fun))
    
    mlp_mfa_plus_index.append(mfa_plus_index)
    mlp_mfa_plus_scores.append(-res_gp.fun)
    mlp_mfa_plus_hidden_layer_sizes.append(res_gp.x[0])
    mlp_mfa_plus_alphas.append(res_gp.x[1])

best_mlp_mfa_plus_index = mlp_mfa_plus_scores.index(max(mlp_mfa_plus_scores))

mlp_mfa_plus_dict = {"mlp_mfa_plus_index": mlp_mfa_plus_index,
                     "mlp_mfa_plus_scores": mlp_mfa_plus_scores,
                     "mlp_mfa_plus_hidden_layer_sizes": mlp_mfa_plus_hidden_layer_sizes,
                     "mlp_mfa_plus_alphas": mlp_mfa_plus_alphas,
                     "best_mlp_mfa_plus_index": best_mlp_mfa_plus_index}
outputPickleFileHandle = open("mlp_mfa_plus_dict.pkl", "wb")
pickle.dump(mlp_mfa_plus_dict, outputPickleFileHandle)
outputPickleFileHandle.close()



















