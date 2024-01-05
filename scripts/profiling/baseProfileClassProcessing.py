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
Classes={0:'Files',1:'Browsing',2:'Images', 3:'Streaming'}
plt.ion()
nfig=1

## -- 2 -- ##
features_files=np.loadtxt("./normal_file_uploadingAllF.dat")
features_browsing=np.loadtxt("./normal_google_browsingAllF.dat")
features_images=np.loadtxt("./normal_image_uploadingAllF.dat")
features_streaming=np.loadtxt("./normal_zoom_webcamAllF.dat")


oClass_files=np.ones((len(features_files),1))*0
oClass_browsing=np.ones((len(features_browsing),1))*1
oClass_images=np.ones((len(features_images),1))*2
oClass_streaming=np.ones((len(features_streaming),1))*3


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


i4train=np.vstack((trainFeatures_files,trainFeatures_browsing, trainFeatures_images, trainFeatures_streaming))
o4trainClass=np.vstack((oClass_files[:pFiles],oClass_browsing[:pBrowse], oClass_images[:pImages], oClass_streaming[:pStream]))

# #:ii For classification
i4Ctrain=np.vstack((trainFeatures_files,trainFeatures_browsing, trainFeatures_images, trainFeatures_streaming))
o4trainClass=np.vstack((oClass_files[:pFiles],oClass_browsing[:pBrowse], oClass_images[:pImages], oClass_streaming[:pStream]))

# #:iii For anomaly detection and classification using last 50% of data
testFeatures_files=features_files[pFiles:,:]
testFeatures_browsing=features_browsing[pBrowse:,:]
testFeatures_images=features_images[pImages:,:]
testFeatures_streaming=features_streaming[pStream:,:]


i4Atest=np.vstack((testFeatures_files,testFeatures_browsing,testFeatures_images,testFeatures_streaming))
o4testClass=np.vstack((oClass_files[pFiles:],oClass_browsing[pBrowse:],oClass_images[pImages:],oClass_streaming[pStream:]))

## -- 4 -- ##
print('\n-- Clustering with K-Means --')
kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto")
i4Ctrain=np.vstack((trainFeatures_files,trainFeatures_browsing,trainFeatures_images,trainFeatures_streaming))
#i3Ctrain = StandardScaler().fit_transform(i3Ctrain)
labels= kmeans.fit_predict(i4Ctrain)

for i in range(len(labels)):
    print('Obs: {:2} ({}): K-Means Cluster Label: -> {}'.format(i,Classes[o4testClass[i][0]],labels[i]))
    
## -- 5 -- ##
print('\n-- Clustering with DBSCAN --')
i4Ctrain=np.vstack((trainFeatures_files,trainFeatures_browsing,trainFeatures_images,trainFeatures_streaming))
i4Ctrain = StandardScaler().fit_transform(i4Ctrain)
db = DBSCAN(eps=0.5, min_samples=10).fit(i4Ctrain)
labels = db.labels_


for i in range(len(labels)):
    print('Obs: {:2} ({}): DBSCAN Cluster Label: -> {}'.format(i,Classes[o4testClass[i][0]],labels[i]))

sys.exit(0)

## -- 7 -- ##

i2train=np.vstack((trainFeatures_browsing,trainFeatures_yt))
#scaler = MaxAbsScaler().fit(i2train)
#i2train=scaler.transform(i2train)

centroids={}
for c in range(2):  # Only the first two classes
    pClass=(o2trainClass==c).flatten()
    centroids.update({c:np.mean(i2train[pClass,:],axis=0)})
print('All Features Centroids:\n',centroids)

i3Atest=np.vstack((testFeatures_browsing,testFeatures_yt,testFeatures_mining))
#i3Atest=scaler.transform(i3Atest)

AnomalyThreshold=10

print('\n-- Anomaly Detection based on Centroids Distances --')
nObsTest,nFea=i3Atest.shape
for i in range(nObsTest):
    x=i3Atest[i]
    dists=[distance(x,centroids[0]),distance(x,centroids[1])]
    if min(dists)>AnomalyThreshold:
        result="Anomaly"
    else:
        result="OK"
       
    print('Obs: {:2} ({}): Normalized Distances to Centroids: [{:.4f},{:.4f}] -> Result -> {}'.format(i,Classes[o3testClass[i][0]],*dists,result))


## -- 8 -- ##

print('\n-- Anomaly Detection based on One Class Support Vector Machines--')
i2train=np.vstack((trainFeatures_browsing,trainFeatures_yt))
i3Atest=np.vstack((testFeatures_browsing,testFeatures_yt,testFeatures_mining))

nu=0.1
ocsvm = svm.OneClassSVM(gamma='scale',kernel='linear',nu=nu).fit(i2train)  
rbf_ocsvm = svm.OneClassSVM(gamma='scale',kernel='rbf',nu=nu).fit(i2train)  
poly_ocsvm = svm. OneClassSVM(gamma='scale',kernel='poly',nu=nu,degree=2).fit(i2train)  

L1=ocsvm.predict(i3Atest)
L2=rbf_ocsvm.predict(i3Atest)
L3=poly_ocsvm.predict(i3Atest)

AnomResults={-1:"Anomaly",1:"OK"}

true_pos=0
false_pos=0

nObsTest,nFea=i3Atest.shape
for i in range(nObsTest):
    if (L1[i]==-1 and L2[i]==-1 and L3[i]==-1):
        if (o3testClass[i][0] == 2.0):
            true_pos+=1
        else:
            false_pos+=1
    print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[o3testClass[i][0]],AnomResults[L1[i]],AnomResults[L2[i]],AnomResults[L3[i]]))

print('\nTrue Positives: {} | False Positives: {}'.format(true_pos,false_pos))
print('Precision: {:.4f} | Recall: {:.4f}'.format(true_pos/(true_pos+false_pos)*100,true_pos/pM))

## -- 10 -- #
print('\n-- Classification based on Support Vector Machines --')

i3train=np.vstack((trainFeatures_browsing,trainFeatures_yt,trainFeatures_mining))
i3Ctest=np.vstack((testFeatures_browsing,testFeatures_yt,testFeatures_mining))


svc = svm.SVC(kernel='linear').fit(i3train, o3trainClass)  
rbf_svc = svm.SVC(kernel='rbf').fit(i3train, o3trainClass)  
poly_svc = svm.SVC(kernel='poly',degree=2).fit(i3train, o3trainClass)  

L1=svc.predict(i3Ctest)
L2=rbf_svc.predict(i3Ctest)
L3=poly_svc.predict(i3Ctest)
print('\n')

linear_true_pos = 0
linear_false_pos = 0
rbf_true_pos = 0
rbf_false_pos = 0
poly_true_pos = 0
poly_false_pos = 0

nObsTest,nFea=i3Ctest.shape
for i in range(nObsTest):
    if (L1[i]==o3testClass[i][0]):
        linear_true_pos+=1
    else:
        linear_false_pos+=1
    if (L2[i]==o3testClass[i][0]):
        rbf_true_pos+=1
    else:
        rbf_false_pos+=1
    if (L3[i]==o3testClass[i][0]):
        poly_true_pos+=1
    else:
        poly_false_pos+=1
    print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[o3testClass[i][0]],Classes[L1[i]],Classes[L2[i]],Classes[L3[i]]))

print('\nLinear True Positives: {} | False Positives: {}'.format(linear_true_pos,linear_false_pos))
print('Linear Precision: {:.4f} | Recall: {:.4f}'.format(linear_true_pos/(linear_true_pos+linear_false_pos)*100,linear_true_pos/pM))
print('\nRBF True Positives: {} | False Positives: {}'.format(rbf_true_pos,rbf_false_pos))
print('RBF Precision: {:.4f} | Recall: {:.4f}'.format(rbf_true_pos/(rbf_true_pos+rbf_false_pos)*100,rbf_true_pos/pM))
print('\nPoly True Positives: {} | False Positives: {}'.format(poly_true_pos,poly_false_pos))
print('Poly Precision: {:.4f} | Recall: {:.4f}'.format(poly_true_pos/(poly_true_pos+poly_false_pos)*100,poly_true_pos/pM))

sys.exit(0)
## -- 12 -- ##
from sklearn.neural_network import MLPClassifier
print('\n-- Classification based on Neural Networks --')
i3train=np.vstack((trainFeatures_browsing,trainFeatures_yt,trainFeatures_mining))
i3Ctest=np.vstack((testFeatures_browsing,testFeatures_yt,testFeatures_mining))

scaler = MaxAbsScaler().fit(i3train)
i3trainN=scaler.transform(i3train)
i3CtestN=scaler.transform(i3Ctest)


alpha=1
max_iter=100000
clf = MLPClassifier(solver='lbfgs',alpha=alpha,hidden_layer_sizes=(20,),max_iter=max_iter)
clf.fit(i3trainN, o3trainClass) 
LT=clf.predict(i3CtestN) 

nObsTest,nFea=i3CtestN.shape
for i in range(nObsTest):
    print('Obs: {:2} ({:<8}): Classification->{}'.format(i,Classes[o3testClass[i][0]],Classes[LT[i]]))
