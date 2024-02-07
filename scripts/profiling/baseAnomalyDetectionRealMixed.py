import copy
import numpy as np
import scipy.stats as stats
import scipy.signal as signal
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from scipy.stats import multivariate_normal
from sklearn import svm
import time
import sys
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

def plot_confusion(true_positive, false_positive, true_negative, false_negative, title):
    conf_matrix = np.array([[true_positive, false_negative], [false_positive, true_negative]])
    fig, ax = plt.subplots()

    threshold = conf_matrix.max() / 2
    cax = ax.matshow(conf_matrix, cmap=plt.cm.Blues)
    plt.title(title)
    fig.colorbar(cax)

    for (i, j), val in np.ndenumerate(conf_matrix):
        color = 'white' if conf_matrix[i, j] > threshold else 'black'
        ax.text(j, i, '{:0.2f}'.format(val), ha='center', va='center', color=color)

    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_xticklabels([''] + ['Anomaly', 'OK'])
    ax.set_yticklabels([''] + ['Anomaly', 'OK'])

    plt.show(block=True)

########### Main Code #############
Classes={0:'Normal',1:'Rat'}
plt.ion()
nfig=1

## -- 2 -- ##


# features_normal=np.loadtxt("./45min_good-UA_mixed1.0AllF.dat")
# features_mixed = np.loadtxt("./45min_all_mixed1.0AllF.dat")

features_normal=np.loadtxt("./45min_good-UA_mixed0.1AllF.dat")
features_mixed = np.loadtxt("./45min_all_mixed0.1AllF.dat")


oClass_normal=np.ones((len(features_normal),1))*0
oClass_mixed=np.ones((len(features_mixed),1))*1


features=np.vstack((features_normal,features_mixed))
oClass=np.vstack((oClass_normal,oClass_mixed))

#TRAIN
percentage=0.75
pNormal=int(len(features_normal)*percentage)
trainFeatures_Normal=features_normal[0:pNormal,:]

#TEST
# mixed_percentage=0.25
mixed_percentage=1.0   
pMixed=int(len(features_mixed)*mixed_percentage)

testFeatures_normal=features_normal[pNormal:,:]
testFeatures_mixed=features_mixed[0:pMixed,:]

oTestClass=np.vstack((oClass_normal[pNormal:,:],oClass_mixed[0:pMixed,:]))

print('\n-- Anomaly Detection based on One Class Support Vector Machines--')
i4train=np.vstack((trainFeatures_Normal))
i5Atest=np.vstack((testFeatures_normal,testFeatures_mixed))

nu=0.1
ocsvm = svm.OneClassSVM(gamma='scale',kernel='linear',nu=nu).fit(i4train)  
rbf_ocsvm = svm.OneClassSVM(gamma='scale',kernel='rbf',nu=nu).fit(i4train)  
poly_ocsvm = svm. OneClassSVM(gamma='scale',kernel='poly',nu=nu,degree=2).fit(i4train)  

L1=ocsvm.predict(i5Atest)
L2=rbf_ocsvm.predict(i5Atest)
L3=poly_ocsvm.predict(i5Atest)

AnomResults={-1:"Anomaly",1:"OK"}

linear_true_pos=0
linear_false_pos=0
linear_true_neg=0
linear_false_neg=0

rbf_true_pos=0
rbf_false_pos=0
rbf_true_neg=0
rbf_false_neg=0

poly_true_pos=0
poly_false_pos=0
poly_true_neg=0
poly_false_neg=0

nObsTest,nFea=i5Atest.shape
for i in range(nObsTest):

    #Just for RBF
    if (L2[i]==-1 and oTestClass[i][0] == 1):
        rbf_true_pos+=1
    elif (L2[i]==-1 and oTestClass[i][0] != 1):
        rbf_false_pos+=1
    elif (L2[i]==1 and oTestClass[i][0] == 1):
        rbf_false_neg+=1
    elif (L2[i]==1 and oTestClass[i][0] != 1):
        rbf_true_neg+=1


    # print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[oTestClass[i][0]],AnomResults[L1[i]],AnomResults[L2[i]],AnomResults[L3[i]]))


logger.info("\n-- RBF Kernel --")
print('\nTrue Positives: {} | False Positives: {} | False Negatives: {} | True Negatives: {}'.format(rbf_true_pos,rbf_false_pos,rbf_false_neg,rbf_true_neg))
#percentages
true_pos_perc = 0 if (rbf_true_pos+rbf_false_neg) == 0 else rbf_true_pos/(rbf_true_pos+rbf_false_neg)
false_pos_perc=0 if (rbf_true_neg+rbf_false_pos) == 0 else rbf_false_pos/(rbf_true_neg+rbf_false_pos)
false_neg_perc=0 if (rbf_false_neg+rbf_true_pos) == 0 else rbf_false_neg/(rbf_false_neg+rbf_true_pos)
true_neg_perc=0 if (rbf_false_pos+rbf_true_neg) == 0 else rbf_true_neg/(rbf_false_pos+rbf_true_neg)
print('True Positives: {:.4f} | False Positives: {:.4f} | False Negatives: {:.4f} | True Negatives: {:.4f}'.format(true_pos_perc,false_pos_perc,false_neg_perc,true_neg_perc))

#metrics
print("\nMetrics")
accu = 0 if (rbf_true_pos+rbf_true_neg+rbf_false_pos+rbf_false_neg) == 0 else (rbf_true_pos+rbf_true_neg)/(rbf_true_pos+rbf_true_neg+rbf_false_pos+rbf_false_neg)
prec = 0 if (rbf_true_pos+rbf_false_pos) == 0 else rbf_true_pos/(rbf_true_pos+rbf_false_pos)
rec = 0 if (rbf_true_pos+rbf_false_neg) == 0 else rbf_true_pos/(rbf_true_pos+rbf_false_neg)
f1 = 0 if (prec+rec) == 0 else 2*(prec*rec)/(prec+rec)
print("Accuracy: {:.4f}".format(accu))
print("Precision: {:.4f}".format(prec))
print("Recall: {:.4f}".format(rec))
print("F1 Score: {:.4f}".format(f1))
plot_confusion(true_pos_perc, false_pos_perc, true_neg_perc, false_neg_perc, "RBF Kernel")

# plot_confusion(true_pos_perc, false_pos_perc, true_neg_perc, false_neg_perc, "RBF Kernel")


from sklearn.mixture import GaussianMixture
logger.info('\n-- Anomaly Detection based on Gaussian Mixture Models --')
i4train=np.vstack((trainFeatures_Normal))
i5Atest=np.vstack((testFeatures_normal,testFeatures_mixed))

gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=0).fit(i4train)
log_likelihood=gmm.score_samples(i5Atest)

threshold=-1000

anomalies_gm = log_likelihood < threshold

true_pos=0
false_pos=0
true_neg=0
false_neg=0

nObsTest,nFea=i5Atest.shape
for i in range(nObsTest):
    anomaly_status = 'Anomaly' if anomalies_gm[i] else 'OK'

    if (anomaly_status=="Anomaly" and oTestClass[i][0] == 1):
        true_pos+=1
    elif (anomaly_status=="Anomaly" and oTestClass[i][0] != 1):
        false_pos+=1
    elif (anomaly_status=="OK" and oTestClass[i][0] == 1):
        false_neg+=1
    elif (anomaly_status=="OK" and oTestClass[i][0] != 1):
        true_neg+=1

    # print('Obs: {:2} ({:<8}): GMM Result: {}'.format(i,Classes[oTestClass[i][0]],anomaly_status))

print('\nTrue Positives: {} | False Positives: {} | False Negatives: {} | True Negatives: {}'.format(true_pos,false_pos,false_neg,true_neg))
#percentages
print('True Positives: {:.4f} | False Positives: {:.4f} | False Negatives: {:.4f} | True Negatives: {:.4f}'.format(true_pos/(true_pos+false_neg)*100,false_pos/(true_neg+false_pos)*100,false_neg/(false_neg+true_pos)*100,true_neg/(false_pos+true_neg)*100))

#metrics
print("\nMetrics")
print("Accuracy: {:.4f}".format((true_pos+true_neg)/(true_pos+true_neg+false_pos+false_neg)))
print("Precision: {:.4f}".format(true_pos/(true_pos+false_pos)))
print("Recall: {:.4f}".format(true_pos/(true_pos+false_neg)))
print("F1 Score: {:.4f}".format(2*(true_pos/(true_pos+false_pos))*(true_pos/(true_pos+false_neg))/((true_pos/(true_pos+false_pos))+(true_pos/(true_pos+false_neg)))))
plot_confusion(true_pos/(true_pos+false_neg), false_pos/(true_neg+false_pos), true_neg/(false_pos+true_neg), false_neg/(false_neg+true_pos), "GMM")

logger.info('\n-- Anomaly Detection based on Isolation Forest--')
from sklearn.ensemble import IsolationForest

# Set parameters for Isolation Forest
n_estimators = 100  # Number of trees in the forest
contamination = 0.1  # Proportion of outliers in the data set

# Create the model
iforest = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)

# Fit the model and predict
iforest.fit(i4train)  # Unlike LOF, IF has a separate fit method
if_labels = iforest.predict(i5Atest)

# Convert labels to a more readable format
if_results = {1: "OK", -1: "Anomaly"}
if_predictions = [if_results[label] for label in if_labels]

true_pos=0
false_pos=0
true_neg=0
false_neg=0

for i in range(len(if_predictions)):
    if (if_predictions[i]=="Anomaly" and oTestClass[i][0] == 1):
        true_pos+=1
    elif (if_predictions[i]=="Anomaly" and oTestClass[i][0] != 1):
        false_pos+=1
    elif (if_predictions[i]=="OK" and oTestClass[i][0] == 1):
        false_neg+=1
    elif (if_predictions[i]=="OK" and oTestClass[i][0] != 1):
        true_neg+=1

    # print('Obs: {:2} ({:<8}): IF Result: {}'.format(i,Classes[o5testClass[i][0]],if_predictions[i]))

print('\nTrue Positives: {} | False Positives: {} | False Negatives: {} | True Negatives: {}'.format(true_pos,false_pos,false_neg,true_neg))
#percentages
print('True Positives: {:.4f} | False Positives: {:.4f} | False Negatives: {:.4f} | True Negatives: {:.4f}'.format(true_pos/(true_pos+false_neg)*100,false_pos/(true_neg+false_pos)*100,false_neg/(false_neg+true_pos)*100,true_neg/(false_pos+true_neg)*100))

#metrics
print("\nMetrics")
print("Accuracy: {:.4f}".format((true_pos+true_neg)/(true_pos+true_neg+false_pos+false_neg)))
print("Precision: {:.4f}".format(true_pos/(true_pos+false_pos)))
print("Recall: {:.4f}".format(true_pos/(true_pos+false_neg)))
print("F1 Score: {:.4f}".format(2*(true_pos/(true_pos+false_pos))*(true_pos/(true_pos+false_neg))/((true_pos/(true_pos+false_pos))+(true_pos/(true_pos+false_neg)))))
plot_confusion(true_pos/(true_pos+false_neg), false_pos/(true_neg+false_pos), true_neg/(false_pos+true_neg), false_neg/(false_neg+true_pos), "Isolation Forest")

# Ensemble - Bagging
logger.info('\n-- Anomaly Detection based on Ensemble Bagging--')
from sklearn.ensemble import BaggingClassifier

# Train each model
iforest = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42).fit(i4train)
gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=0).fit(i4train)
# rbf = svm.OneClassSVM(gamma='scale',kernel='rbf',nu=nu).fit(i4train)

# Make predictions
iforest_decisions = iforest.predict(i5Atest)
gmm_scores = gmm.score_samples(i5Atest)
# rbf_decisions = rbf.predict(i5Atest)
rbf_decisions = copy.deepcopy(L2)

# Normalize GMM scores to be between -1 and 1, assuming that lower scores are more anomalous
threshold=-1000

anomalies_gm = log_likelihood < threshold
anomalies_gm_labels = np.where(anomalies_gm, -1, 1)

# Combine predictions
combine_predictions = []
for i in range(len(i5Atest)):
    votes = anomalies_gm_labels[i] + iforest_decisions[i] + rbf_decisions[i]
    # votes = anomalies_gm_labels[i] + rbf_decisions[i]
    combine_prediction = -1 if votes < 0 else 1
    combine_predictions.append(combine_prediction)

# Convert labels to a more readable format
ensemble_results = {1: "OK", -1: "Anomaly"}
combine_predictions = [ensemble_results[label] for label in combine_predictions]

true_pos=0
false_pos=0
true_neg=0
false_neg=0

for i in range(len(combine_predictions)):
    if (combine_predictions[i]=="Anomaly" and oTestClass[i][0] == 1):
        true_pos+=1
    elif (combine_predictions[i]=="Anomaly" and oTestClass[i][0] != 1):
        false_pos+=1
    elif (combine_predictions[i]=="OK" and oTestClass[i][0] == 1):
        false_neg+=1
    elif (combine_predictions[i]=="OK" and oTestClass[i][0] != 1):
        true_neg+=1

    # print('Obs: {:2} ({:<8}): IF Result: {}'.format(i,Classes[o5testClass[i][0]],if_predictions[i]))
        
print('\nTrue Positives: {} | False Positives: {} | False Negatives: {} | True Negatives: {}'.format(true_pos,false_pos,false_neg,true_neg))
#percentages
print('True Positives: {:.4f} | False Positives: {:.4f} | False Negatives: {:.4f} | True Negatives: {:.4f}'.format(true_pos/(true_pos+false_neg)*100,false_pos/(true_neg+false_pos)*100,false_neg/(false_neg+true_pos)*100,true_neg/(false_pos+true_neg)*100))

#metrics
print("\nMetrics")
print("Accuracy: {:.4f}".format((true_pos+true_neg)/(true_pos+true_neg+false_pos+false_neg)))
print("Precision: {:.4f}".format(true_pos/(true_pos+false_pos)))
print("Recall: {:.4f}".format(true_pos/(true_pos+false_neg)))
print("F1 Score: {:.4f}".format(2*(true_pos/(true_pos+false_pos))*(true_pos/(true_pos+false_neg))/((true_pos/(true_pos+false_pos))+(true_pos/(true_pos+false_neg)))))
plot_confusion(true_pos/(true_pos+false_neg), false_pos/(true_neg+false_pos), true_neg/(false_pos+true_neg), false_neg/(false_neg+true_pos), "Ensemble Bagging")

# # Ensemble - Bayes Optimal Classifier
# logger.info('\n-- Anomaly Detection based on Ensemble Bayes Optimal Classifier--')

# from scipy.special import expit  # Sigmoid function

# # Train the Isolation Forest model
# iforest = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)
# iforest.fit(i4train)
# # Obtain decision scores
# iforest_scores = iforest.decision_function(i5Atest)
# # Convert to probabilities using the sigmoid function
# iforest_probs = expit(iforest_scores)
# # print("IForest Probs: ", iforest_probs)

# # Train the Gaussian Mixture Model
# gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=0)
# gmm.fit(i4train)
# # Get log probabilities
# gmm_log_probs = gmm.score_samples(i5Atest)
# # Convert log probabilities to true probabilities
# gmm_probs = np.exp(gmm_log_probs)
# # print("GMM Probs: ", gmm_probs)

# # Assume rbf_decisions are obtained from the One-Class SVM
# # Normalize SVM decision scores to [0, 1] using the sigmoid function
# svm_probs = expit(rbf_decisions)
# # print("SVM Probs: ", svm_probs)

# # Average the probabilities from each model
# # Here, you can also apply different weights to each model's output if desired
# average_probs = (iforest_probs + gmm_probs + svm_probs) / 3
# # print("Average Probs: ", average_probs)

# # This threshold could be tuned based on cross-validation on a validation set
# threshold = 0.5

# # Make final decision based on the average probabilities
# final_decisions = np.where(average_probs < threshold, -1, 1)

# # Calculate confusion matrix elements
# tp = np.sum((final_decisions == -1) & (oTestClass == 1))
# fp = np.sum((final_decisions == -1) & (oTestClass == 0))
# tn = np.sum((final_decisions == 1) & (oTestClass == 0))
# fn = np.sum((final_decisions == 1) & (oTestClass == 1))

# # Calculate performance metrics
# accuracy = (tp + tn) / (tp + fp + tn + fn)
# precision = tp / (tp + fp) if (tp + fp) > 0 else 0
# recall = tp / (tp + fn) if (tp + fn) > 0 else 0
# f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# print('\nTrue Positives: {} | False Positives: {} | False Negatives: {} | True Negatives: {}'.format(tp,fp,fn,tn))
# #percentages
# print('True Positives: {:.4f} | False Positives: {:.4f} | False Negatives: {:.4f} | True Negatives: {:.4f}'.format(tp/(tp+fn)*100,fp/(tn+fp)*100,fn/(fn+tp)*100,tn/(fp+tn)*100))

# # Display metrics
# print(f"Accuracy: {accuracy:.4f}")
# print(f"Precision: {precision:.4f}")
# print(f"Recall: {recall:.4f}")
# print(f"F1 Score: {f1_score:.4f}")

# # Assuming you have a function to plot confusion matrix normalized
# plot_confusion(tp/(tp+fn), fp/(tn+fp), tn/(fp+tn), fn/(fn+tp), "Ensemble Bayes Optimal Classifier")