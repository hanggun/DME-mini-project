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

# Split data into train dataset (85%) and test dataset (15%).
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
scoring    = {'precision': precision_scorer, 'recall': recall_scorer, 'f1': f1_scorer, 'accuracy': accuracy_scorer, 'log_loss': log_loss_scorer}
cv_results = cross_validate(dummy_classifier_most_frequent, X_train_sc, y_train, cv=cv, scoring=scoring)

dummy_classifier_most_frequent_cv_precision = np.mean(cv_results['test_precision'])
dummy_classifier_most_frequent_cv_recall    = np.mean(cv_results['test_recall'])
dummy_classifier_most_frequent_cv_f1        = np.mean(cv_results['test_f1'])
dummy_classifier_most_frequent_cv_accuracy  = np.mean(cv_results['test_accuracy'])
dummy_classifier_most_frequent_cv_log_loss  = np.mean(cv_results['test_log_loss'])

print("Baseline LOOCV precision (Most frequent class): {:.4f}".format(dummy_classifier_most_frequent_cv_precision))    
print("Baseline LOOCV recall (Most frequent class): {:.4f}".format(dummy_classifier_most_frequent_cv_recall))    
print("Baseline LOOCV F1 (Most frequent class): {:.4f}".format(dummy_classifier_most_frequent_cv_f1))    # 0.0000
print("Baseline LOOCV accuracy (Most frequent class): {:.4f}".format(dummy_classifier_most_frequent_cv_accuracy))    # 0.653
print("Baseline LOOCV log-loss (Most frequent class): {}".format(dummy_classifier_most_frequent_cv_log_loss))    # 11.982840790071053

dummy_classifier_random = DummyClassifier(strategy='uniform', random_state=random_state)

cv         = KFold(n_splits=X_train_sc.shape[0], shuffle=True, random_state=random_state)
scoring    = {'precision': precision_scorer, 'recall': recall_scorer, 'f1': f1_scorer, 'accuracy': accuracy_scorer, 'log_loss': log_loss_scorer}
cv_results = cross_validate(dummy_classifier_random, X_train_sc, y_train, cv=cv, scoring=scoring)

dummy_classifier_random_cv_precision = np.mean(cv_results['test_precision'])
dummy_classifier_random_cv_recall    = np.mean(cv_results['test_recall'])
dummy_classifier_random_cv_f1        = np.mean(cv_results['test_f1'])
dummy_classifier_random_cv_accuracy  = np.mean(cv_results['test_accuracy'])
dummy_classifier_random_cv_log_loss  = np.mean(cv_results['test_log_loss'])

print("Baseline LOOCV precision (Random prediction): {:.4f}".format(dummy_classifier_random_cv_precision))    
print("Baseline LOOCV recall (Random prediction): {:.4f}".format(dummy_classifier_random_cv_recall))    
print("Baseline LOOCV F1 (Random prediction): {:.4f}".format(dummy_classifier_random_cv_f1))    # 0.3469
print("Baseline LOOCV accuracy (Random prediction): {:.4f}".format(dummy_classifier_random_cv_accuracy))    # 0.347
print("Baseline LOOCV log-loss (Random prediction): {}".format(dummy_classifier_random_cv_log_loss))    # 22.556457790916486

# TODO: Set back to 30.
# Set feature counts to be considered.
feature_counts = [i + 1 for i in range(50)]

'''
Linear SVM Classifier
'''

# Linear SVM classifier with F-test.
#linear_svm_f_test_scores = []
#linear_svm_f_test_Cs     = []
#linear_svm_f_test_gammas = []
#
#for feature_count in feature_counts:
#    print("Fine tune linear SVM classifier (F-test) with k = {}".format(feature_count))
#    
#    X_train_f_test_sc, X_train_f_test_descriptions, f_test_index = f_test(X_train_sc, y_train, gene_descriptions, feature_count)
#
#    cv = KFold(n_splits=X_train_f_test_sc.shape[0], shuffle=True, random_state=random_state)
#    
#    linear_svm_classifier = SVC(kernel='linear')
#    
#    space = [Real(10**-3, 10**3, "uniform", name="C"),
#             Real(10**-4, 10**1, "uniform", name="gamma")]
#    x0    = [1, 10**-2]
#    
#    @use_named_args(space)
#    def objective_linear_svm(**params):    
#        linear_svm_classifier.set_params(**params)
#        return -np.mean(cross_val_score(linear_svm_classifier, X_train_f_test_sc, y_train, cv=cv, n_jobs=-1, scoring="accuracy"))
#    
#    res_gp = gp_minimize(func=objective_linear_svm, dimensions=space, x0=x0, n_calls=30, random_state=random_state, n_random_starts=5, kappa=1.9)
#    
#    print("Best score = {}".format(-res_gp.fun))
#    
#    linear_svm_f_test_scores.append(-res_gp.fun)
#    linear_svm_f_test_Cs.append(res_gp.x[0])
#    linear_svm_f_test_gammas.append(res_gp.x[1])
#
#best_linear_svm_f_test_index = linear_svm_f_test_scores.index(max(linear_svm_f_test_scores))

# Linear SVM classifier with TNoM.
#linear_svm_tnom_scores = []
#linear_svm_tnom_Cs     = []
#linear_svm_tnom_gammas = []
#
#for feature_count in feature_counts:
#    print("Fine tune linear SVM classifier (TNoM) with k = {}".format(feature_count))
#    
#    X_train_tnom_sc, X_train_tnom_descriptions, tnom_index = TNoM(X_train_sc, y_train, gene_descriptions, feature_count)
#
#    cv = KFold(n_splits=X_train_tnom_sc.shape[0], shuffle=True, random_state=random_state)
#    
#    linear_svm_classifier = SVC(kernel='linear')
#    
#    space = [Real(10**-3, 10**3, "uniform", name="C"),
#             Real(10**-4, 10**1, "uniform", name="gamma")]
#    x0    = [1, 10**-2]
#    
#    @use_named_args(space)
#    def objective_linear_svm(**params):    
#        linear_svm_classifier.set_params(**params)
#        return -np.mean(cross_val_score(linear_svm_classifier, X_train_tnom_sc, y_train, cv=cv, n_jobs=-1, scoring="accuracy"))
#    
#    res_gp = gp_minimize(func=objective_linear_svm, dimensions=space, x0=x0, n_calls=30, random_state=random_state, n_random_starts=5, kappa=1.9)
#    
#    print("Best score = {}".format(-res_gp.fun))
#    
#    linear_svm_tnom_scores.append(-res_gp.fun)
#    linear_svm_tnom_Cs.append(res_gp.x[0])
#    linear_svm_tnom_gammas.append(res_gp.x[1])
#
#best_linear_svm_tnom_index = linear_svm_tnom_scores.index(max(linear_svm_tnom_scores))

# Linear SVM classifier with MFA.
linear_svm_mfa_scores = []
linear_svm_mfa_Cs     = []
linear_svm_mfa_gammas = []

for feature_count in feature_counts:
    print("Fine tune linear SVM classifier (MFA) with k = {}".format(feature_count))
    
    X_train_mfa_sc, X_train_mfa_descriptions, mfa_index = MFA(X_train_sc, y_train, gene_descriptions, feature_count)

    cv = KFold(n_splits=X_train_mfa_sc.shape[0], shuffle=True, random_state=random_state)
    
    linear_svm_classifier = SVC(kernel='linear')
    
    space = [Real(10**-3, 10**3, "uniform", name="C"),
             Real(10**-4, 10**1, "uniform", name="gamma")]
    x0    = [1, 10**-2]
    
    @use_named_args(space)
    def objective_linear_svm(**params):    
        linear_svm_classifier.set_params(**params)
        return -np.mean(cross_val_score(linear_svm_classifier, X_train_mfa_sc, y_train, cv=cv, n_jobs=-1, scoring="accuracy"))
    
    res_gp = gp_minimize(func=objective_linear_svm, dimensions=space, x0=x0, n_calls=30, random_state=random_state, n_random_starts=5, kappa=1.9)
    
    print("Best score = {}".format(-res_gp.fun))
    
    linear_svm_mfa_scores.append(-res_gp.fun)
    linear_svm_mfa_Cs.append(res_gp.x[0])
    linear_svm_mfa_gammas.append(res_gp.x[1])

best_linear_svm_mfa_index = linear_svm_mfa_scores.index(max(linear_svm_mfa_scores))

# Linear SVM classifier with MFA-plus.
#linear_svm_mfa_plus_scores = []
#linear_svm_mfa_plus_Cs     = []
#linear_svm_mfa_plus_gammas = []
#
#for feature_count in feature_counts:
#    print("Fine tune linear SVM classifier (MFA-plus) with k = {}".format(feature_count))
#    
#    X_train_mfa_plus_sc, X_train_mfa_plus_descriptions, mfa_plus_index = MFAplus(X_train_sc, y_train, gene_descriptions, feature_count)
#
#    cv = KFold(n_splits=X_train_mfa_plus_sc.shape[0], shuffle=True, random_state=random_state)
#    
#    linear_svm_classifier = SVC(kernel='linear')
#    
#    space = [Real(10**-3, 10**3, "uniform", name="C"),
#             Real(10**-4, 10**1, "uniform", name="gamma")]
#    x0    = [1, 10**-2]
#    
#    @use_named_args(space)
#    def objective_linear_svm(**params):    
#        linear_svm_classifier.set_params(**params)
#        return -np.mean(cross_val_score(linear_svm_classifier, X_train_mfa_plus_sc, y_train, cv=cv, n_jobs=-1, scoring="accuracy"))
#    
#    res_gp = gp_minimize(func=objective_linear_svm, dimensions=space, x0=x0, n_calls=30, random_state=random_state, n_random_starts=5, kappa=1.9)
#    
#    print("Best score = {}".format(-res_gp.fun))
#    
#    linear_svm_mfa_plus_scores.append(-res_gp.fun)
#    linear_svm_mfa_plus_Cs.append(res_gp.x[0])
#    linear_svm_mfa_plus_gammas.append(res_gp.x[1])
#
#best_linear_svm_mfa_plus_index = linear_svm_mfa_plus_scores.index(max(linear_svm_mfa_plus_scores))

'''
Gaussian Naive Bayes Classifier
'''

# Gaussian Naive Bayes classifier with F-test.
#gaussian_nb_f_test_scores         = []
#gaussian_nb_f_test_var_smoothings = []
#
#for feature_count in feature_counts:
#    print("Fine tune Gaussian Naive Bayes classifier (F-test) with k = {}".format(feature_count))
#
#    X_train_f_test_sc, X_train_f_test_descriptions, f_test_index = f_test(X_train_sc, y_train, gene_descriptions, feature_count)
#
#    cv = KFold(n_splits=X_train_f_test_sc.shape[0], shuffle=True, random_state=random_state)
#    
#    gaussian_nb_classifier = GaussianNB()
#    
#    space = [Real(10**-12, 10**-6, "uniform", name="var_smoothing")]
#    
#    @use_named_args(space)
#    def objective_gaussian_nb(**params):
#        gaussian_nb_classifier.set_params(**params)
#        return -np.mean(cross_val_score(gaussian_nb_classifier, X_train_f_test_sc, y_train, cv=cv, n_jobs=-1, scoring="accuracy"))
#
#    res_gp = gp_minimize(func=objective_gaussian_nb, dimensions=space, n_calls=30, random_state=random_state, n_random_starts=5, kappa=1.9)
#
#    print("Best score = {}".format(-res_gp.fun))
#
#    gaussian_nb_f_test_scores.append(-res_gp.fun)
#    gaussian_nb_f_test_var_smoothings.append(res_gp.x[0])
#
#best_gaussian_nb_f_test_index = gaussian_nb_f_test_scores.index(max(gaussian_nb_f_test_scores))

# Gaussian Naive Bayes classifier with TNoM.
#gaussian_nb_tnom_scores         = []
#gaussian_nb_tnom_var_smoothings = []
#
#for feature_count in feature_counts:
#    print("Fine tune Gaussian Naive Bayes classifier (TNoM) with k = {}".format(feature_count))
#
#    X_train_tnom_sc, X_train_tnom_descriptions, tnom_index = TNoM(X_train_sc, y_train, gene_descriptions, feature_count)
#
#    cv = KFold(n_splits=X_train_tnom_sc.shape[0], shuffle=True, random_state=random_state)
#    
#    gaussian_nb_classifier = GaussianNB()
#    
#    space = [Real(10**-12, 10**-6, "uniform", name="var_smoothing")]
#    
#    @use_named_args(space)
#    def objective_gaussian_nb(**params):
#        gaussian_nb_classifier.set_params(**params)
#        return -np.mean(cross_val_score(gaussian_nb_classifier, X_train_tnom_sc, y_train, cv=cv, n_jobs=-1, scoring="accuracy"))
#
#    res_gp = gp_minimize(func=objective_gaussian_nb, dimensions=space, n_calls=30, random_state=random_state, n_random_starts=5, kappa=1.9)
#
#    print("Best score = {}".format(-res_gp.fun))
#
#    gaussian_nb_tnom_scores.append(-res_gp.fun)
#    gaussian_nb_tnom_var_smoothings.append(res_gp.x[0])
#
#best_gaussian_nb_tnom_index = gaussian_nb_tnom_scores.index(max(gaussian_nb_tnom_scores))

# Gaussian Naive Bayes classifier with MFA.
#gaussian_nb_mfa_scores         = []
#gaussian_nb_mfa_var_smoothings = []
#
#for feature_count in feature_counts:
#    print("Fine tune Gaussian Naive Bayes classifier (MFA) with k = {}".format(feature_count))
#
#    X_train_mfa_sc, X_train_mfa_descriptions, mfa_index = MFA(X_train_sc, y_train, gene_descriptions, feature_count)
#
#    cv = KFold(n_splits=X_train_mfa_sc.shape[0], shuffle=True, random_state=random_state)
#    
#    gaussian_nb_classifier = GaussianNB()
#    
#    space = [Real(10**-12, 10**-6, "uniform", name="var_smoothing")]
#    
#    @use_named_args(space)
#    def objective_gaussian_nb(**params):
#        gaussian_nb_classifier.set_params(**params)
#        return -np.mean(cross_val_score(gaussian_nb_classifier, X_train_mfa_sc, y_train, cv=cv, n_jobs=-1, scoring="accuracy"))
#
#    res_gp = gp_minimize(func=objective_gaussian_nb, dimensions=space, n_calls=30, random_state=random_state, n_random_starts=5, kappa=1.9)
#
#    print("Best score = {}".format(-res_gp.fun))
#
#    gaussian_nb_mfa_scores.append(-res_gp.fun)
#    gaussian_nb_mfa_var_smoothings.append(res_gp.x[0])
#
#best_gaussian_nb_mfa_index = gaussian_nb_mfa_scores.index(max(gaussian_nb_mfa_scores))

# Gaussian Naive Bayes classifier with MFA-plus.
#gaussian_nb_mfa_plus_scores         = []
#gaussian_nb_mfa_plus_var_smoothings = []
#
#for feature_count in feature_counts:
#    print("Fine tune Gaussian Naive Bayes classifier (MFA-plus) with k = {}".format(feature_count))
#
#    X_train_mfa_plus_sc, X_train_mfa_plus_descriptions, mfa_plus_index = MFAplus(X_train_sc, y_train, gene_descriptions, feature_count)
#
#    cv = KFold(n_splits=X_train_mfa_plus_sc.shape[0], shuffle=True, random_state=random_state)
#    
#    gaussian_nb_classifier = GaussianNB()
#    
#    space = [Real(10**-12, 10**-6, "uniform", name="var_smoothing")]
#    
#    @use_named_args(space)
#    def objective_gaussian_nb(**params):
#        gaussian_nb_classifier.set_params(**params)
#        return -np.mean(cross_val_score(gaussian_nb_classifier, X_train_mfa_plus_sc, y_train, cv=cv, n_jobs=-1, scoring="accuracy"))
#
#    res_gp = gp_minimize(func=objective_gaussian_nb, dimensions=space, n_calls=30, random_state=random_state, n_random_starts=5, kappa=1.9)
#
#    print("Best score = {}".format(-res_gp.fun))
#
#    gaussian_nb_mfa_plus_scores.append(-res_gp.fun)
#    gaussian_nb_mfa_plus_var_smoothings.append(res_gp.x[0])
#
#best_gaussian_nb_mfa_plus_index = gaussian_nb_mfa_plus_scores.index(max(gaussian_nb_mfa_plus_scores))

'''
KNN Classifier
'''





# KNN classifier with TNoM.
#knn_tnom_scores      = []
#knn_tnom_n_neighbors = []
#knn_tnom_leaf_sizes  = []
#knn_tnom_ps          = []
#
#for feature_count in feature_counts:
#    print("Fine tune KNN classifier (TNoM) with k = {}".format(feature_count))
#
#    X_train_tnom_sc, X_train_tnom_descriptions, tnom_index = TNoM(X_train_sc, y_train, gene_descriptions, feature_count)
#
#    cv = KFold(n_splits=X_train_tnom_sc.shape[0], shuffle=True, random_state=random_state)
#
#    knn_classifier = KNeighborsClassifier()
#
#    space = [Integer(1, 20, name="n_neighbors"),
#             Integer(5, 55, name="leaf_size"),
#             Integer(1, 10, name="p")]
#
#    @use_named_args(space)
#    def objective_knn(**params):
#        knn_classifier.set_params(**params)
#        return -np.mean(cross_val_score(knn_classifier, X_train_tnom_sc, y_train, cv=cv, n_jobs=-1, scoring="accuracy"))
#    
#    res_gp = gp_minimize(func=objective_knn, dimensions=space, n_calls=30, random_state=random_state, n_random_starts=5, kappa=1.9)
#    
#    print("Best score = {}".format(-res_gp.fun))
#    
#    knn_tnom_scores.append(-res_gp.fun)
#    knn_tnom_n_neighbors.append(res_gp.x[0])
#    knn_tnom_leaf_sizes.append(res_gp.x[1])
#    knn_tnom_ps.append(res_gp.x[2])
#
#best_knn_tnom_index = knn_tnom_scores.index(max(knn_tnom_scores))


'''
Logistic Regression Classifier
'''

# Logistic regression classifier with TNoM.
#logistic_regression_tnom_scores = []
#logistic_regression_tnom_Cs     = []
#
#for feature_count in feature_counts:
#    print("Fine tune logistic regression classifier (TNoM) with k = {}".format(feature_count))
#    
#    X_train_tnom_sc, X_train_tnom_descriptions, tnom_index = TNoM(X_train_sc, y_train, gene_descriptions, feature_count)
#
#    cv = KFold(n_splits=X_train_tnom_sc.shape[0], shuffle=True, random_state=random_state)
#    
#    logistic_regression_classifier = LogisticRegression()
#    
#    space = [Real(1.0, 10**3, "uniform", name="C")]
#    
#    @use_named_args(space)
#    def objective_logistic_regression(**params):
#        logistic_regression_classifier.set_params(**params)
#        return -np.mean(cross_val_score(logistic_regression_classifier, X_train_tnom_sc, y_train, cv=cv, n_jobs=-1, scoring="accuracy"))
#    
#    res_gp = gp_minimize(func=objective_logistic_regression, dimensions=space, n_calls=30, random_state=random_state, n_random_starts=5, kappa=1.9)
#    
#    print("Best score = {}".format(-res_gp.fun))
#    
#    logistic_regression_tnom_scores.append(-res_gp.fun)
#    logistic_regression_tnom_Cs.append(res_gp.x[0])
#
#best_logistic_regression_tnom_index = logistic_regression_tnom_scores.index(max(logistic_regression_tnom_scores))





'''
Multi-Layer Perceptron Classifier
'''

# Multi-layer perceptron classifier with TNoM.
#mlp_tnom_scores             = []
#mlp_tnom_hidden_layer_sizes = []
#mlp_tnom_alphas             = []
#
#for feature_count in feature_counts:
#    print("Fine tune multi-layer perceptron classifier (TNoM) with k = {}".format(feature_count))
#    
#    X_train_tnom_sc, X_train_tnom_descriptions, tnom_index = TNoM(X_train_sc, y_train, gene_descriptions, feature_count)
#
#    cv = KFold(n_splits=X_train_tnom_sc.shape[0], shuffle=True, random_state=random_state)
#    
#    mlp_classifier = MLPClassifier(random_state=random_state)
#    
#    space = [Integer(10, 1000, name="hidden_layer_sizes"),
#             Real(10**-8, 1, name="alpha")]
#    
#    @use_named_args(space)
#    def objective_mlp(**params):    
#        mlp_classifier.set_params(**params)
#        return -np.mean(cross_val_score(mlp_classifier, X_train_tnom_sc, y_train, cv=cv, n_jobs=-1, scoring="accuracy"))
#    
#    res_gp = gp_minimize(func=objective_mlp, dimensions=space, n_calls=30, random_state=random_state, n_random_starts=5, kappa=1.9)
#
#    print("Best score = {}".format(-res_gp.fun))
#    
#    mlp_tnom_scores.append(-res_gp.fun)
#    mlp_tnom_hidden_layer_sizes.append(res_gp.x[0])
#    mlp_tnom_alphas.append(res_gp.x[1])
#
#best_mlp_tnom_index = mlp_tnom_scores.index(max(mlp_tnom_scores))


















# TEMP:
#svc_opt = SVC(kernel='linear',
#             C=res_gp.x[0],
#             gamma=res_gp.x[1]).fit(X_train_tnom_sc,y_train)
#print("Classification accuracy on validation set: {:.3f}".format(accuracy_score(y_val, svc_opt.predict(X_val_tnom_sc))))

# TEMP:
#temp_data = [0.8979591836734694,
#             0.8979591836734694,
#             0.8979591836734694,
#             0.8979591836734694,
#             0.8979591836734694,
#             0.8571428571428571,
#             0.8775510204081632,
#             0.8979591836734694,
#             0.9183673469387755,
#             0.8979591836734694,
#             0.8775510204081632,
#             0.8979591836734694,
#             0.8571428571428571,
#             0.8775510204081632,
#             0.8775510204081632,
#             0.8571428571428571,
#             0.8367346938775511,
#             0.8979591836734694,
#             0.8571428571428571,
#             0.8367346938775511,
#             0.8571428571428571,
#             0.8979591836734694,
#             0.8979591836734694,
#             0.8163265306122449,
#             0.8571428571428571,
#             0.8367346938775511,
#             0.8367346938775511,
#             0.8775510204081632,
#             0.8775510204081632,
#             0.8775510204081632]
#fig = plt.figure()
#x = np.array(feature_counts)
#y = np.array(temp_data)
#plt.plot(x, y)
#plt.title('Linear Support Vector Classifier (TNoM)')
#plt.xlabel('Number of genes')
#plt.ylabel('LOOCV accuracy')
#fig.tight_layout()
#plt.show()












# TODO: Consider removing precision, recall, and F1 scores from baseline.

# TODO: Consider PCA, kernel PCA, F-test, MFA, MFA-plus.


















