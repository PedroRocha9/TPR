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
Classes={0:'Files',1:'Browsing',2:'Images', 3:'Streaming', 4:'Rat'}
plt.ion()
nfig=1

## -- 2 -- ##
features_files=np.loadtxt("./55min_filesAll30F.dat")
features_browsing=np.loadtxt("./52min_browsingAll30F.dat")
features_images=np.loadtxt("./51min_imageAll30F.dat")
features_streaming=np.loadtxt("./50min_zoomAll30F.dat")
features_rat=np.loadtxt("./45min_ratV1All30F.dat")


oClass_files=np.ones((len(features_files),1))*0
oClass_browsing=np.ones((len(features_browsing),1))*1
oClass_images=np.ones((len(features_images),1))*2
oClass_streaming=np.ones((len(features_streaming),1))*3
oClass_rat=np.ones((len(features_rat),1))*4


features=np.vstack((features_files,features_browsing,features_images,features_streaming))
oClass=np.vstack((oClass_files,oClass_browsing,oClass_images,oClass_streaming))

## -- 3 -- ##
#:i - For anomaly detection
percentage=0.5
pFiles=int(len(features_files)*percentage)
trainFeatures_files=features_files

pBrowse=int(len(features_browsing)*percentage)
trainFeatures_browsing=features_browsing

pImages=int(len(features_images)*percentage)
trainFeatures_images=features_images

pStream=int(len(features_streaming)*percentage)
trainFeatures_streaming=features_streaming

rat_percentage=0.5
pRat=int(len(features_rat)*rat_percentage)

# #:iii For anomaly detection and classification using last 50% of data
testFeatures_files=features_files
testFeatures_browsing=features_browsing
testFeatures_images=features_images
testFeatures_streaming=features_streaming
testFeatures_rat=features_rat[:pRat,:]



i5Atest=np.vstack((testFeatures_files,testFeatures_browsing,testFeatures_images,testFeatures_streaming, testFeatures_rat))
o5testClass=np.vstack((oClass_files,oClass_browsing,oClass_images,oClass_streaming, oClass_rat[:pRat,:]))

print('\n-- Anomaly Detection based on One Class Support Vector Machines--')
i4train=np.vstack((trainFeatures_files,trainFeatures_browsing,trainFeatures_images,trainFeatures_streaming))
i5Atest=np.vstack((testFeatures_rat))

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
       #Just for Linear
    if (L1[i]==-1):
        linear_true_pos+=1
    elif (L1[i]==1):
        linear_false_neg+=1

    #Just for RBF
    if (L2[i]==-1):
        rbf_true_pos+=1
    elif (L2[i]==1):
        rbf_false_neg+=1

    #Just for Poly
    if (L3[i]==-1):
        poly_true_pos+=1
    elif (L3[i]==1):
        poly_false_neg+=1

    # print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,"RAT",AnomResults[L1[i]],AnomResults[L2[i]],AnomResults[L3[i]]))
    
# logger.info("\n-- Linear Kernel --")
# print('\nTrue Positives: {} | False Positives: {} | False Negatives: {} | True Negatives: {}'.format(linear_true_pos,linear_false_pos,linear_false_neg,linear_true_neg))
# #percentages
# true_pos_perc = 0 if (linear_true_pos+linear_false_neg) == 0 else linear_true_pos/(linear_true_pos+linear_false_neg)*100
# false_pos_perc=0 if (linear_true_neg+linear_false_pos) == 0 else linear_false_pos/(linear_true_neg+linear_false_pos)*100
# false_neg_perc=0 if (linear_false_neg+linear_true_pos) == 0 else linear_false_neg/(linear_false_neg+linear_true_pos)*100
# true_neg_perc=0 if (linear_false_pos+linear_true_neg) == 0 else linear_true_neg/(linear_false_pos+linear_true_neg)*100
# print('True Positives: {:.4f} | False Positives: {:.4f} | False Negatives: {:.4f} | True Negatives: {:.4f}'.format(true_pos_perc,false_pos_perc,false_neg_perc,true_neg_perc))

# #metrics
# print("\nMetrics")
# accu = 0 if (linear_true_pos+linear_true_neg+linear_false_pos+linear_false_neg) == 0 else (linear_true_pos+linear_true_neg)/(linear_true_pos+linear_true_neg+linear_false_pos+linear_false_neg)
# prec = 0 if (linear_true_pos+linear_false_pos) == 0 else linear_true_pos/(linear_true_pos+linear_false_pos)
# rec = 0 if (linear_true_pos+linear_false_neg) == 0 else linear_true_pos/(linear_true_pos+linear_false_neg)
# f1 = 0 if (prec+rec) == 0 else 2*(prec*rec)/(prec+rec)
# print("Accuracy: {:.4f}".format(accu))
# print("Precision: {:.4f}".format(prec))
# print("Recall: {:.4f}".format(rec))
# print("F1 Score: {:.4f}".format(f1))

logger.info("\n-- RBF Kernel --")
# print('\nTrue Positives: {} | False Positives: {} | False Negatives: {} | True Negatives: {}'.format(rbf_true_pos,rbf_false_pos,rbf_false_neg,rbf_true_neg))
#percentages
true_pos_perc = 0 if (rbf_true_pos+rbf_false_neg) == 0 else rbf_true_pos/(rbf_true_pos+rbf_false_neg)*100
false_pos_perc=0 if (rbf_true_neg+rbf_false_pos) == 0 else rbf_false_pos/(rbf_true_neg+rbf_false_pos)*100
false_neg_perc=0 if (rbf_false_neg+rbf_true_pos) == 0 else rbf_false_neg/(rbf_false_neg+rbf_true_pos)*100
true_neg_perc=0 if (rbf_false_pos+rbf_true_neg) == 0 else rbf_true_neg/(rbf_false_pos+rbf_true_neg)*100
# print('True Positives: {:.4f} | False Positives: {:.4f} | False Negatives: {:.4f} | True Negatives: {:.4f}'.format(true_pos_perc,false_pos_perc,false_neg_perc,true_neg_perc))


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
plot_confusion(true_pos_perc/100, false_pos_perc/100, true_neg_perc/100, false_neg_perc/100, "RBF Kernel")



# logger.info("\n-- Poly Kernel --")
# print('\nTrue Positives: {} | False Positives: {} | False Negatives: {} | True Negatives: {}'.format(poly_true_pos,poly_false_pos,poly_false_neg,poly_true_neg))
# #percentages
# true_pos_perc = 0 if (poly_true_pos+poly_false_neg) == 0 else poly_true_pos/(poly_true_pos+poly_false_neg)*100
# false_pos_perc=0 if (poly_true_neg+poly_false_pos) == 0 else poly_false_pos/(poly_true_neg+poly_false_pos)*100
# false_neg_perc=0 if (poly_false_neg+poly_true_pos) == 0 else poly_false_neg/(poly_false_neg+poly_true_pos)*100
# true_neg_perc=0 if (poly_false_pos+poly_true_neg) == 0 else poly_true_neg/(poly_false_pos+poly_true_neg)*100
# print('True Positives: {:.4f} | False Positives: {:.4f} | False Negatives: {:.4f} | True Negatives: {:.4f}'.format(true_pos_perc,false_pos_perc,false_neg_perc,true_neg_perc))


# #metrics
# print("\nMetrics")
# accu = 0 if (poly_true_pos+poly_true_neg+poly_false_pos+poly_false_neg) == 0 else (poly_true_pos+poly_true_neg)/(poly_true_pos+poly_true_neg+poly_false_pos+poly_false_neg)
# prec = 0 if (poly_true_pos+poly_false_pos) == 0 else poly_true_pos/(poly_true_pos+poly_false_pos)
# rec = 0 if (poly_true_pos+poly_false_neg) == 0 else poly_true_pos/(poly_true_pos+poly_false_neg)
# f1 = 0 if (prec+rec) == 0 else 2*(prec*rec)/(prec+rec)
# print("Accuracy: {:.4f}".format(accu))
# print("Precision: {:.4f}".format(prec))
# print("Recall: {:.4f}".format(rec))
# print("F1 Score: {:.4f}".format(f1))


## -- Anomaly Detection based on Gaussian Mixture Models -- ##


from sklearn.mixture import GaussianMixture
logger.info('\n-- Anomaly Detection based on Gaussian Mixture Models --')
i4train=np.vstack((trainFeatures_files,trainFeatures_browsing,trainFeatures_images,trainFeatures_streaming))
i5Atest=np.vstack((testFeatures_rat))

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

    if (anomaly_status=="Anomaly"):
        true_pos+=1
    elif (anomaly_status=="OK"):
        false_neg+=1

    # print('Obs: {:2} ({:<8}): GMM Result: {}'.format(i,"RAT",anomaly_status))

# print('\nTrue Positives: {} | False Positives: {} | False Negatives: {} | True Negatives: {}'.format(true_pos,false_pos,false_neg,true_neg))
#percentages
true_pos_perc = 0 if (true_pos+false_neg) == 0 else true_pos/(true_pos+false_neg)*100
false_pos_perc=0 if (true_neg+false_pos) == 0 else false_pos/(true_neg+false_pos)*100
false_neg_perc=0 if (false_neg+true_pos) == 0 else false_neg/(false_neg+true_pos)*100
true_neg_perc=0 if (false_pos+true_neg) == 0 else true_neg/(false_pos+true_neg)*100
# print('True Positives: {:.4f} | False Positives: {:.4f} | False Negatives: {:.4f} | True Negatives: {:.4f}'.format(true_pos_perc,false_pos_perc,false_neg_perc,true_neg_perc))


#metrics
print("\nMetrics")
accu = 0 if (true_pos+true_neg+false_pos+false_neg) == 0 else (true_pos+true_neg)/(true_pos+true_neg+false_pos+false_neg)
prec = 0 if (true_pos+false_pos) == 0 else true_pos/(true_pos+false_pos)
rec = 0 if (true_pos+false_neg) == 0 else true_pos/(true_pos+false_neg)
f1 = 0 if (prec+rec) == 0 else 2*(prec*rec)/(prec+rec)
print("Accuracy: {:.4f}".format(accu))
print("Precision: {:.4f}".format(prec))
print("Recall: {:.4f}".format(rec))
print("F1 Score: {:.4f}".format(f1))
plot_confusion(true_pos_perc/100, false_pos_perc/100, true_neg_perc/100, false_neg_perc/100, "GMM")


## -- Anomaly Detection based on Isolation Forest-- ##

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
    if (if_predictions[i]=="Anomaly"):
        true_pos+=1
    elif (if_predictions[i]=="OK"):
        false_neg+=1

    # print('Obs: {:2} ({:<8}): IF Result: {}'.format(i,"RAT",if_predictions[i]))

# print('\nTrue Positives: {} | False Positives: {} | False Negatives: {} | True Negatives: {}'.format(true_pos,false_pos,false_neg,true_neg))
#percentages
true_pos_perc = 0 if (true_pos+false_neg) == 0 else true_pos/(true_pos+false_neg)*100
false_pos_perc=0 if (true_neg+false_pos) == 0 else false_pos/(true_neg+false_pos)*100
false_neg_perc=0 if (false_neg+true_pos) == 0 else false_neg/(false_neg+true_pos)*100
true_neg_perc=0 if (false_pos+true_neg) == 0 else true_neg/(false_pos+true_neg)*100

# print('True Positives: {:.4f} | False Positives: {:.4f} | False Negatives: {:.4f} | True Negatives: {:.4f}'.format(true_pos_perc,false_pos_perc,false_neg_perc,true_neg_perc))

#metrics
print("\nMetrics")
accu = 0 if (true_pos+true_neg+false_pos+false_neg) == 0 else (true_pos+true_neg)/(true_pos+true_neg+false_pos+false_neg)
prec = 0 if (true_pos+false_pos) == 0 else true_pos/(true_pos+false_pos)
rec = 0 if (true_pos+false_neg) == 0 else true_pos/(true_pos+false_neg)
f1 = 0 if (prec+rec) == 0 else 2*(prec*rec)/(prec+rec)
print("Accuracy: {:.4f}".format(accu))
print("Precision: {:.4f}".format(prec))
print("Recall: {:.4f}".format(rec))
print("F1 Score: {:.4f}".format(f1))
plot_confusion(true_pos_perc/100, false_pos_perc/100, true_neg_perc/100, false_neg_perc/100, "Isolation Forest")

