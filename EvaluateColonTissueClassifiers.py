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
from sklearn.metrics import confusion_matrix
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

# Invert test labels. Now, +1 means tumor tissue, -1 means normal tissue. Later, classifer outputs will also be inverted
# to match with the inverted test labels. Inversion is used here because we want to calculate precision, recall, and F1
# scores against the tumor tissues, which we want ultimately care about.
y_test_inverted = y_test * -1

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

# Read training results.
inputFileHandle = open("Models/linear_svm_f_test_dict.pkl", "rb")
linear_svm_f_test_dict = pickle.load(inputFileHandle)

inputFileHandle = open("Models/linear_svm_tnom_dict.pkl", "rb")
linear_svm_tnom_dict = pickle.load(inputFileHandle)

inputFileHandle = open("Models/linear_svm_mfa_dict.pkl", "rb")
linear_svm_mfa_dict = pickle.load(inputFileHandle)

inputFileHandle = open("Models/linear_svm_mfa_plus_dict.pkl", "rb")
linear_svm_mfa_plus_dict = pickle.load(inputFileHandle)

inputFileHandle = open("Models/gaussian_nb_f_test_dict.pkl", "rb")
gaussian_nb_f_test_dict = pickle.load(inputFileHandle)

inputFileHandle = open("Models/gaussian_nb_tnom_dict.pkl", "rb")
gaussian_nb_tnom_dict = pickle.load(inputFileHandle)

inputFileHandle = open("Models/gaussian_nb_mfa_dict.pkl", "rb")
gaussian_nb_mfa_dict = pickle.load(inputFileHandle)

#inputFileHandle = open("Models/gaussian_nb_mfa_plus_dict.pkl", "rb")
#gaussian_nb_mfa_plus_dict = pickle.load(inputFileHandle)

inputFileHandle = open("Models/knn_f_test_dict.pkl", "rb")
knn_f_test_dict = pickle.load(inputFileHandle)

inputFileHandle = open("Models/knn_tnom_dict.pkl", "rb")
knn_tnom_dict = pickle.load(inputFileHandle)

#inputFileHandle = open("Models/knn_mfa_dict.pkl", "rb")
#knn_mfa_dict = pickle.load(inputFileHandle)
#
#inputFileHandle = open("Models/knn_mfa_plus_dict.pkl", "rb")
#knn_mfa_plus_dict = pickle.load(inputFileHandle)

inputFileHandle = open("Models/logistic_regression_f_test_dict.pkl", "rb")
logistic_regression_f_test_dict = pickle.load(inputFileHandle)

inputFileHandle = open("Models/logistic_regression_tnom_dict.pkl", "rb")
logistic_regression_tnom_dict = pickle.load(inputFileHandle)

#inputFileHandle = open("Models/logistic_regression_mfa_dict.pkl", "rb")
#logistic_regression_mfa_dict = pickle.load(inputFileHandle)
#
#inputFileHandle = open("Models/logistic_regression_mfa_plus_dict.pkl", "rb")
#logistic_regression_mfa_plus_dict = pickle.load(inputFileHandle)

inputFileHandle = open("Models/mlp_f_test_dict.pkl", "rb")
mlp_f_test_dict = pickle.load(inputFileHandle)

inputFileHandle = open("Models/mlp_tnom_dict.pkl", "rb")
mlp_tnom_dict = pickle.load(inputFileHandle)

#inputFileHandle = open("Models/mlp_mfa_dict.pkl", "rb")
#mlp_mfa_dict = pickle.load(inputFileHandle)
#
#inputFileHandle = open("Models/mlp_mfa_plus_dict.pkl", "rb")
#mlp_mfa_plus_dict = pickle.load(inputFileHandle)

# Plot graphs.
feature_counts = [i + 1 for i in range(50)]

fig = plt.figure()
x = np.array(feature_counts)
y1 = np.array(linear_svm_f_test_dict["linear_svm_f_test_scores"])
y2 = np.array(linear_svm_tnom_dict["linear_svm_tnom_scores"])
y3 = np.array(linear_svm_mfa_dict["linear_svm_mfa_scores"])
y4 = np.array(linear_svm_mfa_plus_dict["linear_svm_mfa_plus_scores"])
plt.plot(x, y1, alpha=0.7, label="F-test")
plt.plot(x, y2, alpha=0.7, label="TNoM score")
plt.plot(x, y3, alpha=0.7, label="MFA score")
plt.plot(x, y4, alpha=0.7, label="MFA score+")
plt.title('Linear SVM Classifier')
plt.xlabel('Number of genes')
plt.ylabel('LOOCV accuracy')
plt.legend(loc='best')
fig.tight_layout()
plt.savefig("Figures/LinearSvmLoocvAccuracy.pdf", dpi=400)


fig = plt.figure()
x = np.array(feature_counts)
y1 = np.array(gaussian_nb_f_test_dict["gaussian_nb_f_test_scores"])
y2 = np.array(gaussian_nb_tnom_dict["gaussian_nb_tnom_scores"])
y3 = np.array(gaussian_nb_mfa_dict["gaussian_nb_mfa_scores"])
#y4 = np.array(gaussian_nb_mfa_plus_dict["gaussian_nb_mfa_plus_scores"])
plt.plot(x, y1, alpha=0.7, label="F-test")
plt.plot(x, y2, alpha=0.7, label="TNoM score")
plt.plot(x, y3, alpha=0.7, label="MFA score")
#plt.plot(x, y4, alpha=0.7, label="MFA score+")
plt.title('Gaussian Naive Bayes Classifier')
plt.xlabel('Number of genes')
plt.ylabel('LOOCV accuracy')
plt.legend(loc='best')
fig.tight_layout()
plt.savefig("Figures/GaussianNaiveBayesLoocvAccuracy.pdf", dpi=400)

fig = plt.figure()
x = np.array(feature_counts)
y1 = np.array(knn_f_test_dict["knn_f_test_scores"])
y2 = np.array(knn_tnom_dict["knn_tnom_scores"])
#y3 = np.array(knn_mfa_dict["knn_mfa_scores"])
#y4 = np.array(knn_mfa_plus_dict["knn_mfa_plus_scores"])
plt.plot(x, y1, alpha=0.7, label="F-test")
plt.plot(x, y2, alpha=0.7, label="TNoM score")
#plt.plot(x, y3, alpha=0.7, label="MFA score")
#plt.plot(x, y4, alpha=0.7, label="MFA score+")
plt.title('KNN Classifier')
plt.xlabel('Number of genes')
plt.ylabel('LOOCV accuracy')
plt.legend(loc='best')
fig.tight_layout()
plt.savefig("Figures/KnnLoocvAccuracy.pdf", dpi=400)

fig = plt.figure()
x = np.array(feature_counts)
y1 = np.array(logistic_regression_f_test_dict["logistic_regression_f_test_scores"])
y2 = np.array(logistic_regression_tnom_dict["logistic_regression_tnom_scores"])
#y3 = np.array(logistic_regression_mfa_dict["logistic_regression_mfa_scores"])
#y4 = np.array(logistic_regression_mfa_plus_dict["logistic_regression_mfa_plus_scores"])
plt.plot(x, y1, alpha=0.7, label="F-test")
plt.plot(x, y2, alpha=0.7, label="TNoM score")
#plt.plot(x, y3, alpha=0.7, label="MFA score")
#plt.plot(x, y4, alpha=0.7, label="MFA score+")
plt.title('Logistic Regression Classifier')
plt.xlabel('Number of genes')
plt.ylabel('LOOCV accuracy')
plt.legend(loc='best')
fig.tight_layout()
plt.savefig("Figures/LogisticRegressionLoocvAccuracy.pdf", dpi=400)

fig = plt.figure()
x = np.array(feature_counts)
y1 = np.array(mlp_f_test_dict["mlp_f_test_scores"])
y2 = np.array(mlp_tnom_dict["mlp_tnom_scores"])
#y3 = np.array(mlp_mfa_dict["mlp_mfa_scores"])
#y4 = np.array(mlp_mfa_plus_dict["mlp_mfa_plus_scores"])
plt.plot(x, y1, alpha=0.7, label="F-test")
plt.plot(x, y2, alpha=0.7, label="TNoM score")
#plt.plot(x, y3, alpha=0.7, label="MFA score")
#plt.plot(x, y4, alpha=0.7, label="MFA score+")
plt.title('Multi-Layer Perceptron Classifier')
plt.xlabel('Number of genes')
plt.ylabel('LOOCV accuracy')
plt.legend(loc='best')
fig.tight_layout()
plt.savefig("Figures/MlpLoocvAccuracy.pdf", dpi=400)

# Evaluate best models on test dataset.

'''
Best MLP Classifier
'''
print("Best MLP classifier with TNoM:")
best_mlp_tnom_index = mlp_tnom_dict["best_mlp_tnom_index"]

X_train_tnom_sc = X_train_sc[:, mlp_tnom_dict["mlp_tnom_index"][best_mlp_tnom_index]]
X_test_tnom_sc  = X_test_sc[:, mlp_tnom_dict["mlp_tnom_index"][best_mlp_tnom_index]]
print("TNoM selected genes:", mlp_tnom_dict["mlp_tnom_index"][best_mlp_tnom_index])

best_mlp_classifier = MLPClassifier(hidden_layer_sizes=mlp_tnom_dict["mlp_tnom_hidden_layer_sizes"][best_mlp_tnom_index],
                                    alpha=mlp_tnom_dict["mlp_tnom_alphas"][best_mlp_tnom_index],
                                    random_state=random_state)
best_mlp_classifier.fit(X_train_tnom_sc, y_train)
best_mlp_classifier_test_pred          = best_mlp_classifier.predict(X_test_tnom_sc)
best_mlp_classifier_test_inverted_pred = best_mlp_classifier_test_pred * -1
best_mlp_classifier_test_pred_proba    = best_mlp_classifier.predict_proba(X_test_tnom_sc)

print("Classification accuracy on test dataset: {:.4f}".format(accuracy_score(y_test_inverted, best_mlp_classifier_test_inverted_pred)))
print("Log-loss score on test dataset : {}".format(log_loss(y_test, best_mlp_classifier_test_pred_proba)))
print("Precision score on test dataset: {:.4f}".format(precision_score(y_test_inverted, best_mlp_classifier_test_inverted_pred)))
print("Recall score on test dataset: {:.4f}".format(recall_score(y_test_inverted, best_mlp_classifier_test_inverted_pred)))
print("F1 score on test dataset: {:.4f}".format(f1_score(y_test_inverted, best_mlp_classifier_test_inverted_pred)))

cm = confusion_matrix(y_test, best_mlp_classifier_test_pred)
fig = plt.figure()
plot_confusion_matrix(cm, normalize=False,
                      classes=["Tumor tissue", "Normal tissue"],
                      title="Confusion Matrix For Best Multi-Layer Perceptron Classifier")
fig.tight_layout()
plt.savefig("Figures/ConfusionMatrixForBestMlpClassifier.pdf", dpi=400)

print("\n")

'''
Best Linear SVM Classifier
'''
print("Best linear SVM classifier with F-test:")
best_linear_svm_f_test_index = linear_svm_f_test_dict["best_linear_svm_f_test_index"]

X_train_f_test_sc = X_train_sc[:, linear_svm_f_test_dict["linear_svm_f_test_index"][best_linear_svm_f_test_index]]
X_test_f_test_sc  = X_test_sc[:, linear_svm_f_test_dict["linear_svm_f_test_index"][best_linear_svm_f_test_index]]
print("F-test selected genes:", linear_svm_f_test_dict["linear_svm_f_test_index"][best_linear_svm_f_test_index])

best_linear_svm_classifier = SVC(kernel='linear',
                                 C=linear_svm_f_test_dict["linear_svm_f_test_Cs"][best_linear_svm_f_test_index],
                                 gamma=linear_svm_f_test_dict["linear_svm_f_test_gammas"][best_linear_svm_f_test_index],
                                 probability=True)

best_linear_svm_classifier.fit(X_train_f_test_sc, y_train)
best_linear_svm_classifier_test_pred          = best_linear_svm_classifier.predict(X_test_f_test_sc)
best_linear_svm_classifier_test_inverted_pred = best_linear_svm_classifier_test_pred * -1
best_linear_svm_classifier_test_pred_proba    = best_linear_svm_classifier.predict_proba(X_test_f_test_sc)

print("Classification accuracy on test dataset: {:.4f}".format(accuracy_score(y_test_inverted, best_linear_svm_classifier_test_inverted_pred)))
print("Log-loss score on test dataset : {}".format(log_loss(y_test, best_linear_svm_classifier_test_pred_proba)))
print("Precision score on test dataset: {:.4f}".format(precision_score(y_test_inverted, best_linear_svm_classifier_test_inverted_pred)))
print("Recall score on test dataset: {:.4f}".format(recall_score(y_test_inverted, best_linear_svm_classifier_test_inverted_pred)))
print("F1 score on test dataset: {:.4f}".format(f1_score(y_test_inverted, best_linear_svm_classifier_test_inverted_pred)))

cm = confusion_matrix(y_test, best_linear_svm_classifier_test_pred)
fig = plt.figure()
plot_confusion_matrix(cm, normalize=False,
                      classes=["Tumor tissue", "Normal tissue"],
                      title="Confusion Matrix For Best Linear SVM Classifier")
fig.tight_layout()
plt.savefig("Figures/ConfusionMatrixForBestLinearSvmClassifier.pdf", dpi=400)

print("\n")

'''
Best Gaussian Naive Bayes Classifier
'''
print("Best Gaussian Naive Bayes classifier with F-test:")
best_gaussian_nb_f_test_index = gaussian_nb_f_test_dict["best_gaussian_nb_f_test_index"]

X_train_f_test_sc = X_train_sc[:, gaussian_nb_f_test_dict["gaussian_nb_f_test_index"][best_gaussian_nb_f_test_index]]
X_test_f_test_sc  = X_test_sc[:, gaussian_nb_f_test_dict["gaussian_nb_f_test_index"][best_gaussian_nb_f_test_index]]
print("F-test selected genes:", gaussian_nb_f_test_dict["gaussian_nb_f_test_index"][best_gaussian_nb_f_test_index])

best_gaussian_nb_classifier = GaussianNB(var_smoothing=gaussian_nb_f_test_dict["gaussian_nb_f_test_var_smoothings"][best_gaussian_nb_f_test_index])

best_gaussian_nb_classifier.fit(X_train_f_test_sc, y_train)
best_gaussian_nb_classifier_test_pred          = best_gaussian_nb_classifier.predict(X_test_f_test_sc)
best_gaussian_nb_classifier_test_inverted_pred = best_gaussian_nb_classifier_test_pred * -1
best_gaussian_nb_classifier_test_pred_proba    = best_gaussian_nb_classifier.predict_proba(X_test_f_test_sc)

print("Classification accuracy on test dataset: {:.4f}".format(accuracy_score(y_test_inverted, best_gaussian_nb_classifier_test_inverted_pred)))
print("Log-loss score on test dataset : {}".format(log_loss(y_test, best_gaussian_nb_classifier_test_pred_proba)))
print("Precision score on test dataset: {:.4f}".format(precision_score(y_test_inverted, best_gaussian_nb_classifier_test_inverted_pred)))
print("Recall score on test dataset: {:.4f}".format(recall_score(y_test_inverted, best_gaussian_nb_classifier_test_inverted_pred)))
print("F1 score on test dataset: {:.4f}".format(f1_score(y_test_inverted, best_gaussian_nb_classifier_test_inverted_pred)))

cm = confusion_matrix(y_test, best_gaussian_nb_classifier_test_pred)
fig = plt.figure()
plot_confusion_matrix(cm, normalize=False,
                      classes=["Tumor tissue", "Normal tissue"],
                      title="Confusion Matrix For Best Gaussian Naive Bayes Classifier")
fig.tight_layout()
plt.savefig("Figures/ConfusionMatrixForBestGaussianNaiveBayesClassifier.pdf", dpi=400)

print("\n")

'''
Best Logistic Regression Classifier
'''
print("Best logistic regression classifier with F-test:")
best_logistic_regression_f_test_index = logistic_regression_f_test_dict["best_logistic_regression_f_test_index"]

X_train_f_test_sc = X_train_sc[:, logistic_regression_f_test_dict["logistic_regression_f_test_index"][best_logistic_regression_f_test_index]]
X_test_f_test_sc  = X_test_sc[:, logistic_regression_f_test_dict["logistic_regression_f_test_index"][best_logistic_regression_f_test_index]]
print("F-test selected genes:", logistic_regression_f_test_dict["logistic_regression_f_test_index"][best_logistic_regression_f_test_index])

best_logistic_regression_classifier = LogisticRegression(C=logistic_regression_f_test_dict["logistic_regression_f_test_Cs"][best_logistic_regression_f_test_index])

best_logistic_regression_classifier.fit(X_train_f_test_sc, y_train)
best_logistic_regression_classifier_test_pred          = best_logistic_regression_classifier.predict(X_test_f_test_sc)
best_logistic_regression_classifier_test_inverted_pred = best_logistic_regression_classifier_test_pred * -1
best_logistic_regression_classifier_test_pred_proba    = best_logistic_regression_classifier.predict_proba(X_test_f_test_sc)

print("Classification accuracy on test dataset: {:.4f}".format(accuracy_score(y_test_inverted, best_logistic_regression_classifier_test_inverted_pred)))
print("Log-loss score on test dataset : {}".format(log_loss(y_test, best_logistic_regression_classifier_test_pred_proba)))
print("Precision score on test dataset: {:.4f}".format(precision_score(y_test_inverted, best_logistic_regression_classifier_test_inverted_pred)))
print("Recall score on test dataset: {:.4f}".format(recall_score(y_test_inverted, best_logistic_regression_classifier_test_inverted_pred)))
print("F1 score on test dataset: {:.4f}".format(f1_score(y_test_inverted, best_logistic_regression_classifier_test_inverted_pred)))

cm = confusion_matrix(y_test, best_logistic_regression_classifier_test_pred)
fig = plt.figure()
plot_confusion_matrix(cm, normalize=False,
                      classes=["Tumor tissue", "Normal tissue"],
                      title="Confusion Matrix For Logistic Regression Classifier")
fig.tight_layout()
plt.savefig("Figures/ConfusionMatrixForBestLogisticRegressionClassifier.pdf", dpi=400)

print("\n")

'''
Best KNN Classifier
'''
print("Best KNN classifier with F-test:")
best_knn_f_test_index = knn_f_test_dict["best_knn_f_test_index"]

X_train_f_test_sc = X_train_sc[:, knn_f_test_dict["knn_f_test_index"][best_knn_f_test_index]]
X_test_f_test_sc  = X_test_sc[:, knn_f_test_dict["knn_f_test_index"][best_knn_f_test_index]]
print("F-test selected genes:", knn_f_test_dict["knn_f_test_index"][best_knn_f_test_index])

best_knn_classifier = KNeighborsClassifier(n_neighbors=knn_f_test_dict["knn_f_test_n_neighbors"][best_knn_f_test_index],
                                           leaf_size=knn_f_test_dict["knn_f_test_leaf_sizes"][best_knn_f_test_index],
                                           p=knn_f_test_dict["knn_f_test_ps"][best_knn_f_test_index])

best_knn_classifier.fit(X_train_f_test_sc, y_train)
best_knn_classifier_test_pred          = best_knn_classifier.predict(X_test_f_test_sc)
best_knn_classifier_test_inverted_pred = best_knn_classifier_test_pred * -1
best_knn_classifier_test_pred_proba    = best_knn_classifier.predict_proba(X_test_f_test_sc)

print("Classification accuracy on test dataset: {:.4f}".format(accuracy_score(y_test_inverted, best_knn_classifier_test_inverted_pred)))
print("Log-loss score on test dataset : {}".format(log_loss(y_test, best_knn_classifier_test_pred_proba)))
print("Precision score on test dataset: {:.4f}".format(precision_score(y_test_inverted, best_knn_classifier_test_inverted_pred)))
print("Recall score on test dataset: {:.4f}".format(recall_score(y_test_inverted, best_knn_classifier_test_inverted_pred)))
print("F1 score on test dataset: {:.4f}".format(f1_score(y_test_inverted, best_knn_classifier_test_inverted_pred)))

cm = confusion_matrix(y_test, best_knn_classifier_test_pred)
fig = plt.figure()
plot_confusion_matrix(cm, normalize=False,
                      classes=["Tumor tissue", "Normal tissue"],
                      title="Confusion Matrix For Best KNN Classifier")
fig.tight_layout()
plt.savefig("Figures/ConfusionMatrixForBestKnnClassifier.pdf", dpi=400)

print("\n")


















