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
import warnings
warnings.filterwarnings('ignore')


def waitforEnter(fstop=False):
    if fstop:
        if sys.version_info[0] == 2:
            raw_input("Press ENTER to continue.")
        else:
            input("Press ENTER to continue.")
            
## -- 3 -- ##
def plotFeatures(features,oClass,f1index=0,f2index=1):
    nObs,nFea=features.shape
    colors=['b','g','r']
    for i in range(nObs):
        plt.plot(features[i,f1index],features[i,f2index],'o'+colors[int(oClass[i])])

    plt.show()
    waitforEnter()
    
def logplotFeatures(features,oClass,f1index=0,f2index=1):
    nObs,nFea=features.shape
    colors=['b','g','r']
    for i in range(nObs):
        plt.loglog(features[i,f1index],features[i,f2index],'o'+colors[int(oClass[i])])

    plt.show()
    waitforEnter()
    
## -- 11 -- ##
def distance(c,p):
    s=0
    n=0
    for i in range(len(c)):
        if c[i]>0:
            s+=np.square((p[i]-c[i])/c[i])
            n+=1
    
    return(np.sqrt(s/n))
        
    #return(np.sqrt(np.sum(np.square((p-c)/c))))

########### Main Code #############
Classes={0:'Files',1:'Browsing',2:'Images', 3:'Streaming', 4:'Rat'}
plt.ion()
nfig=1

## -- 2 -- ##
features_files=np.loadtxt("./55min_filesAllF.dat")
features_browsing=np.loadtxt("./52min_browsingAllF.dat")
features_images=np.loadtxt("./45min_imageAllF.dat")
features_streaming=np.loadtxt("./50min_zoomAllF.dat")
features_rat=np.loadtxt("./45min_ratAllF.dat")


oClass_files=np.ones((len(features_files),1))*0
oClass_browsing=np.ones((len(features_browsing),1))*1
oClass_images=np.ones((len(features_images),1))*2
oClass_streaming=np.ones((len(features_streaming),1))*3
oClass_rat=np.ones((len(features_rat),1))*4


features=np.vstack((features_files,features_browsing,features_images,features_streaming))
oClass=np.vstack((oClass_files,oClass_browsing,oClass_images,oClass_streaming))
# features=np.vstack((features_yt,features_browsing,features_mining))
# oClass=np.vstack((oClass_yt,oClass_browsing,oClass_mining))

# print('Train Silence Features Size:',features.shape)
# plt.figure(2)
# plotFeatures(features,oClass,4,10)
# plt.figure(3)
# plotFeatures(features,oClass,0,7)
# plt.figure(4)
# plotFeatures(features,oClass,2,8)

## -- 3 -- ##
#:i - For anomaly detection
percentage=0.5
pFiles=int(len(features_files)*percentage)
trainFeatures_files=features_files[:pFiles,:]

pBrowse=int(len(features_browsing)*percentage)
trainFeatures_browsing=features_browsing[:pBrowse,:]

pImages=int(len(features_images)*percentage)
trainFeatures_images=features_images[:pImages,:]

pStream=int(len(features_streaming)*percentage)
trainFeatures_streaming=features_streaming[:pStream,:]

pRat=int(len(features_rat)*percentage)
trainFeatures_rat=features_rat[:pRat,:]


i4train=np.vstack((trainFeatures_files,trainFeatures_browsing, trainFeatures_images, trainFeatures_streaming))
o4trainClass=np.vstack((oClass_files[:pFiles],oClass_browsing[:pBrowse], oClass_images[:pImages], oClass_streaming[:pStream]))

# #:ii For classification
i5Ctrain=np.vstack((trainFeatures_files,trainFeatures_browsing, trainFeatures_images, trainFeatures_streaming, trainFeatures_rat))
o5trainClass=np.vstack((oClass_files[:pFiles],oClass_browsing[:pBrowse], oClass_images[:pImages], oClass_streaming[:pStream], oClass_rat[:pRat]))

# #:iii For anomaly detection and classification using last 50% of data
testFeatures_files=features_files[pFiles:,:]
testFeatures_browsing=features_browsing[pBrowse:,:]
testFeatures_images=features_images[pImages:,:]
testFeatures_streaming=features_streaming[pStream:,:]
testFeatures_rat=features_rat[pRat:,:]


i5Atest=np.vstack((testFeatures_files,testFeatures_browsing,testFeatures_images,testFeatures_streaming, testFeatures_rat))
o5testClass=np.vstack((oClass_files[pFiles:],oClass_browsing[pBrowse:],oClass_images[pImages:],oClass_streaming[pStream:], oClass_rat[pRat:]))

print('\n-- Anomaly Detection based on One Class Support Vector Machines--')
i4train=np.vstack((trainFeatures_files,trainFeatures_browsing,trainFeatures_images,trainFeatures_streaming))
i5Atest=np.vstack((testFeatures_files,testFeatures_browsing,testFeatures_images,testFeatures_streaming, testFeatures_rat))

nu=0.1
ocsvm = svm.OneClassSVM(gamma='scale',kernel='linear',nu=nu).fit(i4train)  
rbf_ocsvm = svm.OneClassSVM(gamma='scale',kernel='rbf',nu=nu).fit(i4train)  
poly_ocsvm = svm. OneClassSVM(gamma='scale',kernel='poly',nu=nu,degree=2).fit(i4train)  

L1=ocsvm.predict(i5Atest)
L2=rbf_ocsvm.predict(i5Atest)
L3=poly_ocsvm.predict(i5Atest)

AnomResults={-1:"Anomaly",1:"OK"}

true_pos=0
false_pos=0
true_neg=0
false_neg=0

nObsTest,nFea=i5Atest.shape
for i in range(nObsTest):
    #Just for RBF
    if (L2[i]==-1 and o5testClass[i][0] == 4):
        true_pos+=1
    elif (L2[i]==-1 and o5testClass[i][0] != 4):
        false_pos+=1
    elif (L2[i]==1 and o5testClass[i][0] == 4):
        false_neg+=1
    elif (L2[i]==1 and o5testClass[i][0] != 4):
        true_neg+=1

    # print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[o5testClass[i][0]],AnomResults[L1[i]],AnomResults[L2[i]],AnomResults[L3[i]]))

print('\nTrue Positives: {} | False Positives: {} | False Negatives: {} | True Negatives: {}'.format(true_pos,false_pos,false_neg,true_neg))
#percentages
print('\nTrue Positives: {:.4f} | False Positives: {:.4f} | False Negatives: {:.4f} | True Negatives: {:.4f}'.format(true_pos/(true_pos+false_neg)*100,false_pos/(true_neg+false_pos)*100,false_neg/(false_neg+true_pos)*100,true_neg/(false_pos+true_neg)*100))

#metrics
print("\nMetrics")
print("\nAccuracy: {:.4f}".format((true_pos+true_neg)/(true_pos+true_neg+false_pos+false_neg)))
print("\nPrecision: {:.4f}".format(true_pos/(true_pos+false_pos)))
print("\nRecall: {:.4f}".format(true_pos/(true_pos+false_neg)))
print("\nF1 Score: {:.4f}".format(2*(true_pos/(true_pos+false_pos))*(true_pos/(true_pos+false_neg))/((true_pos/(true_pos+false_pos))+(true_pos/(true_pos+false_neg)))))


## -- 9 -- ##
from sklearn.mixture import GaussianMixture
print('\n-- Anomaly Detection based on Gaussian Mixture Models --')
i4train=np.vstack((trainFeatures_files,trainFeatures_browsing,trainFeatures_images,trainFeatures_streaming))
i5Atest=np.vstack((testFeatures_files,testFeatures_browsing,testFeatures_images,testFeatures_streaming, testFeatures_rat))

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

    if (anomaly_status=="Anomaly" and o5testClass[i][0] == 4):
        true_pos+=1
    elif (anomaly_status=="Anomaly" and o5testClass[i][0] != 4):
        false_pos+=1
    elif (anomaly_status=="OK" and o5testClass[i][0] == 4):
        false_neg+=1
    elif (anomaly_status=="OK" and o5testClass[i][0] != 4):
        true_neg+=1

    # print('Obs: {:2} ({:<8}): GMM Result: {}'.format(i,Classes[o5testClass[i][0]],anomaly_status))

print('\nTrue Positives: {} | False Positives: {} | False Negatives: {} | True Negatives: {}'.format(true_pos,false_pos,false_neg,true_neg))
#percentages
print('\nTrue Positives: {:.4f} | False Positives: {:.4f} | False Negatives: {:.4f} | True Negatives: {:.4f}'.format(true_pos/(true_pos+false_neg)*100,false_pos/(true_neg+false_pos)*100,false_neg/(false_neg+true_pos)*100,true_neg/(false_pos+true_neg)*100))

#metrics
print("\nMetrics")
print("\nAccuracy: {:.4f}".format((true_pos+true_neg)/(true_pos+true_neg+false_pos+false_neg)))
print("\nPrecision: {:.4f}".format(true_pos/(true_pos+false_pos)))
print("\nRecall: {:.4f}".format(true_pos/(true_pos+false_neg)))
print("\nF1 Score: {:.4f}".format(2*(true_pos/(true_pos+false_pos))*(true_pos/(true_pos+false_neg))/((true_pos/(true_pos+false_pos))+(true_pos/(true_pos+false_neg)))))



print('\n-- Anomaly Detection based on Isolation Forest--')
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
    if (if_predictions[i]=="Anomaly" and o5testClass[i][0] == 4):
        true_pos+=1
    elif (if_predictions[i]=="Anomaly" and o5testClass[i][0] != 4):
        false_pos+=1
    elif (if_predictions[i]=="OK" and o5testClass[i][0] == 4):
        false_neg+=1
    elif (if_predictions[i]=="OK" and o5testClass[i][0] != 4):
        true_neg+=1

    # print('Obs: {:2} ({:<8}): IF Result: {}'.format(i,Classes[o5testClass[i][0]],if_predictions[i]))

print('\nTrue Positives: {} | False Positives: {} | False Negatives: {} | True Negatives: {}'.format(true_pos,false_pos,false_neg,true_neg))
#percentages
print('\nTrue Positives: {:.4f} | False Positives: {:.4f} | False Negatives: {:.4f} | True Negatives: {:.4f}'.format(true_pos/(true_pos+false_neg)*100,false_pos/(true_neg+false_pos)*100,false_neg/(false_neg+true_pos)*100,true_neg/(false_pos+true_neg)*100))

#metrics
print("\nMetrics")
print("\nAccuracy: {:.4f}".format((true_pos+true_neg)/(true_pos+true_neg+false_pos+false_neg)))
print("\nPrecision: {:.4f}".format(true_pos/(true_pos+false_pos)))
print("\nRecall: {:.4f}".format(true_pos/(true_pos+false_neg)))
print("\nF1 Score: {:.4f}".format(2*(true_pos/(true_pos+false_pos))*(true_pos/(true_pos+false_neg))/((true_pos/(true_pos+false_pos))+(true_pos/(true_pos+false_neg)))))
