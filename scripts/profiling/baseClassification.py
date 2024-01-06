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
features_files=np.loadtxt("./55min_filesAllF.dat")
features_browsing=np.loadtxt("./52min_browsingAllF.dat")
features_images=np.loadtxt("./45min_imageAllF.dat")
features_streaming=np.loadtxt("./50min_zoomAllF.dat")


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
i4Ctrain = StandardScaler().fit_transform(i4Ctrain)
labels= kmeans.fit_predict(i4Ctrain)

for i in range(len(labels)):
    print('Obs: {:2} ({}): K-Means Cluster Label: -> {}'.format(i,Classes[o4testClass[i][0]],labels[i]))

## -- 5 -- ##
print('\n-- Clustering with DBSCAN --')
i4Ctrain=np.vstack((trainFeatures_files,trainFeatures_browsing,trainFeatures_images,trainFeatures_streaming))
i4Ctrain = StandardScaler().fit_transform(i4Ctrain)
db = DBSCAN(eps=0.1, min_samples=100).fit(i4Ctrain)
labels = db.labels_


for i in range(len(labels)):
    print('Obs: {:2} ({}): DBSCAN Cluster Label: -> {}'.format(i,Classes[o4testClass[i][0]],labels[i]))

sys.exit(0)

## -- 7 -- ## CENTROIDS

# i2train=np.vstack((trainFeatures_browsing,trainFeatures_yt))
# #scaler = MaxAbsScaler().fit(i2train)
# #i2train=scaler.transform(i2train)

# centroids={}
# for c in range(2):  # Only the first two classes
#     pClass=(o2trainClass==c).flatten()
#     centroids.update({c:np.mean(i2train[pClass,:],axis=0)})
# print('All Features Centroids:\n',centroids)

# i3Atest=np.vstack((testFeatures_browsing,testFeatures_yt,testFeatures_mining))
# #i3Atest=scaler.transform(i3Atest)

# AnomalyThreshold=10

# print('\n-- Anomaly Detection based on Centroids Distances --')
# nObsTest,nFea=i3Atest.shape
# for i in range(nObsTest):
#     x=i3Atest[i]
#     dists=[distance(x,centroids[0]),distance(x,centroids[1])]
#     if min(dists)>AnomalyThreshold:
#         result="Anomaly"
#     else:
#         result="OK"
       
#     print('Obs: {:2} ({}): Normalized Distances to Centroids: [{:.4f},{:.4f}] -> Result -> {}'.format(i,Classes[o3testClass[i][0]],*dists,result))




## -- 10 -- #
print('\n-- Classification based on Support Vector Machines --')

i5train=np.vstack((trainFeatures_files,trainFeatures_browsing,trainFeatures_images,trainFeatures_streaming, trainFeatures_rat))
i5Ctest=np.vstack((testFeatures_files,testFeatures_browsing,testFeatures_images,testFeatures_streaming, testFeatures_rat))


print("Training SVMs with Linear")
svc = svm.SVC(kernel='linear').fit(i5train, o5trainClass)  
print("Training SVMs with RBF")
rbf_svc = svm.SVC(kernel='rbf').fit(i5train, o5trainClass)  
print("Training SVMs with Poly")
poly_svc = svm.SVC(kernel='poly',degree=2).fit(i5train, o5trainClass)  

print("Predicting with Linear")
L1=svc.predict(i5Ctest)
print("Predicting with RBF")
L2=rbf_svc.predict(i5Ctest)
print("Predicting with Poly")
L3=poly_svc.predict(i5Ctest)
print('\n')

linear_true_pos = 0
linear_false_pos = 0
rbf_true_pos = 0
rbf_false_pos = 0
poly_true_pos = 0
poly_false_pos = 0

#detailed
# linear_true_pos_files = 0
# linear_false_pos_files = 0
# linear_true_pos_browsing = 0
# linear_false_pos_browsing = 0
# linear_true_pos_images = 0
# linear_false_pos_images = 0
# linear_true_pos_streaming = 0
# linear_false_pos_streaming = 0
# linear_true_pos_rat = 0
# linear_false_pos_rat = 0

# rbf_true_pos_files = 0
# rbf_false_pos_files = 0
# rbf_true_pos_browsing = 0
# rbf_false_pos_browsing = 0
# rbf_true_pos_images = 0
# rbf_false_pos_images = 0
# rbf_true_pos_streaming = 0
# rbf_false_pos_streaming = 0
# rbf_true_pos_rat = 0
# rbf_false_pos_rat = 0

# poly_true_pos_files = 0
# poly_false_pos_files = 0
# poly_true_pos_browsing = 0
# poly_false_pos_browsing = 0
# poly_true_pos_images = 0
# poly_false_pos_images = 0
# poly_true_pos_streaming = 0
# poly_false_pos_streaming = 0
# poly_true_pos_rat = 0
# poly_false_pos_rat = 0

nObsTest,nFea=i5Ctest.shape
for i in range(nObsTest):
    if (L1[i]==o5testClass[i][0]):
        linear_true_pos+=1
    else:
        linear_false_pos+=1
    if (L2[i]==o5testClass[i][0]):
        rbf_true_pos+=1
    else:
        rbf_false_pos+=1
    if (L3[i]==o5testClass[i][0]):
        poly_true_pos+=1
    else:
        poly_false_pos+=1

    #detailed
    # if (L1[i]==o5testClass[i][0] and o5testClass[i][0] == 0):
    #     linear_true_pos_files+=1
    # elif (L1[i]!=o5testClass[i][0] and o5testClass[i][0] == 0):
    #     linear_false_pos_files+=1
    # if (L1[i]==o5testClass[i][0] and o5testClass[i][0] == 1):
    #     linear_true_pos_browsing+=1
    # elif (L1[i]!=o5testClass[i][0] and o5testClass[i][0] == 1):
    #     linear_false_pos_browsing+=1
    # if (L1[i]==o5testClass[i][0] and o5testClass[i][0] == 2):
    #     linear_true_pos_images+=1
    # elif (L1[i]!=o5testClass[i][0] and o5testClass[i][0] == 2):
    #     linear_false_pos_images+=1
    # if (L1[i]==o5testClass[i][0] and o5testClass[i][0] == 3):
    #     linear_true_pos_streaming+=1
    # elif (L1[i]!=o5testClass[i][0] and o5testClass[i][0] == 3):
    #     linear_false_pos_streaming+=1
    # if (L1[i]==o5testClass[i][0] and o5testClass[i][0] == 4):
    #     linear_true_pos_rat+=1
    # elif (L1[i]!=o5testClass[i][0] and o5testClass[i][0] == 4):
    #     linear_false_pos_rat+=1


    # if (L2[i]==o5testClass[i][0] and o5testClass[i][0] == 0):
    #     rbf_true_pos_files+=1
    # elif (L2[i]!=o5testClass[i][0] and o5testClass[i][0] == 0):
    #     rbf_false_pos_files+=1
    # if (L2[i]==o5testClass[i][0] and o5testClass[i][0] == 1):
    #     rbf_true_pos_browsing+=1
    # elif (L2[i]!=o5testClass[i][0] and o5testClass[i][0] == 1):
    #     rbf_false_pos_browsing+=1
    # if (L2[i]==o5testClass[i][0] and o5testClass[i][0] == 2):
    #     rbf_true_pos_images+=1
    # elif (L2[i]!=o5testClass[i][0] and o5testClass[i][0] == 2):
    #     rbf_false_pos_images+=1
    # if (L2[i]==o5testClass[i][0] and o5testClass[i][0] == 3):
    #     rbf_true_pos_streaming+=1
    # elif (L2[i]!=o5testClass[i][0] and o5testClass[i][0] == 3):
    #     rbf_false_pos_streaming+=1
    # if (L2[i]==o5testClass[i][0] and o5testClass[i][0] == 4):
    #     rbf_true_pos_rat+=1

    # if (L3[i]==o5testClass[i][0] and o5testClass[i][0] == 0):
    #     poly_true_pos_files+=1
    # elif (L3[i]!=o5testClass[i][0] and o5testClass[i][0] == 0):
    #     poly_false_pos_files+=1
    # if (L3[i]==o5testClass[i][0] and o5testClass[i][0] == 1):
    #     poly_true_pos_browsing+=1
    # elif (L3[i]!=o5testClass[i][0] and o5testClass[i][0] == 1):
    #     poly_false_pos_browsing+=1
    # if (L3[i]==o5testClass[i][0] and o5testClass[i][0] == 2):
    #     poly_true_pos_images+=1
    # elif (L3[i]!=o5testClass[i][0] and o5testClass[i][0] == 2):
    #     poly_false_pos_images+=1
    # if (L3[i]==o5testClass[i][0] and o5testClass[i][0] == 3):
    #     poly_true_pos_streaming+=1
    # elif (L3[i]!=o5testClass[i][0] and o5testClass[i][0] == 3):
    #     poly_false_pos_streaming+=1
    # if (L3[i]==o5testClass[i][0] and o5testClass[i][0] == 4):
    #     poly_true_pos_rat+=1


    print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[o5testClass[i][0]],Classes[L1[i]],Classes[L2[i]],Classes[L3[i]]))

print('\nLinear True Positives: {} | False Positives: {}'.format(linear_true_pos,linear_false_pos))
print('Linear Precision: {:.4f}'.format(linear_true_pos/(linear_true_pos+linear_false_pos)*100,linear_true_pos))
print('\nRBF True Positives: {} | False Positives: {}'.format(rbf_true_pos,rbf_false_pos))
print('RBF Precision: {:.4f}'.format(rbf_true_pos/(rbf_true_pos+rbf_false_pos)*100))
print('\nPoly True Positives: {} | False Positives: {}'.format(poly_true_pos,poly_false_pos))
print('Poly Precision: {:.4f}'.format(poly_true_pos/(poly_true_pos+poly_false_pos)*100))

# print('\nDetailed')
# print('\nLinear True Positives Files: {} | False Positives Files: {}'.format(linear_true_pos_files,linear_false_pos_files))
# print('Linear Precision Files: {:.4f}'.format(linear_true_pos_files/(linear_true_pos_files+linear_false_pos_files)*100))
# print('\nLinear True Positives Browsing: {} | False Positives Browsing: {}'.format(linear_true_pos_browsing,linear_false_pos_browsing))
# print('Linear Precision Browsing: {:.4f}'.format(linear_true_pos_browsing/(linear_true_pos_browsing+linear_false_pos_browsing)*100))
# print('\nLinear True Positives Images: {} | False Positives Images: {}'.format(linear_true_pos_images,linear_false_pos_images))
# print('Linear Precision Images: {:.4f}'.format(linear_true_pos_images/(linear_true_pos_images+linear_false_pos_images)*100))
# print('\nLinear True Positives Streaming: {} | False Positives Streaming: {}'.format(linear_true_pos_streaming,linear_false_pos_streaming))
# print('Linear Precision Streaming: {:.4f}'.format(linear_true_pos_streaming/(linear_true_pos_streaming+linear_false_pos_streaming)*100))
# print('\nLinear True Positives Rat: {} | False Positives Rat: {}'.format(linear_true_pos_rat,linear_false_pos_rat))
# print('Linear Precision Rat: {:.4f}'.format(linear_true_pos_rat/(linear_true_pos_rat+linear_false_pos_rat)*100))

# print("\n")

# print('\nRBF True Positives Files: {} | False Positives Files: {}'.format(rbf_true_pos_files,rbf_false_pos_files))
# print('RBF Precision Files: {:.4f}'.format(rbf_true_pos_files/(rbf_true_pos_files+rbf_false_pos_files)*100))
# print('\nRBF True Positives Browsing: {} | False Positives Browsing: {}'.format(rbf_true_pos_browsing,rbf_false_pos_browsing))
# print('RBF Precision Browsing: {:.4f}'.format(rbf_true_pos_browsing/(rbf_true_pos_browsing+rbf_false_pos_browsing)*100))
# print('\nRBF True Positives Images: {} | False Positives Images: {}'.format(rbf_true_pos_images,rbf_false_pos_images))
# print('RBF Precision Images: {:.4f}'.format(rbf_true_pos_images/(rbf_true_pos_images+rbf_false_pos_images)*100))
# print('\nRBF True Positives Streaming: {} | False Positives Streaming: {}'.format(rbf_true_pos_streaming,rbf_false_pos_streaming))
# print('RBF Precision Streaming: {:.4f}'.format(rbf_true_pos_streaming/(rbf_true_pos_streaming+rbf_false_pos_streaming)*100))
# print('\nRBF True Positives Rat: {} | False Positives Rat: {}'.format(rbf_true_pos_rat,rbf_false_pos_rat))
# print('RBF Precision Rat: {:.4f}'.format(rbf_true_pos_rat/(rbf_true_pos_rat+rbf_false_pos_rat)*100))

# print("\n")

# print('\nPoly True Positives Files: {} | False Positives Files: {}'.format(poly_true_pos_files,poly_false_pos_files))
# print('Poly Precision Files: {:.4f}'.format(poly_true_pos_files/(poly_true_pos_files+poly_false_pos_files)*100))
# print('\nPoly True Positives Browsing: {} | False Positives Browsing: {}'.format(poly_true_pos_browsing,poly_false_pos_browsing))
# print('Poly Precision Browsing: {:.4f}'.format(poly_true_pos_browsing/(poly_true_pos_browsing+poly_false_pos_browsing)*100))
# print('\nPoly True Positives Images: {} | False Positives Images: {}'.format(poly_true_pos_images,poly_false_pos_images))
# print('Poly Precision Images: {:.4f}'.format(poly_true_pos_images/(poly_true_pos_images+poly_false_pos_images)*100))
# print('\nPoly True Positives Streaming: {} | False Positives Streaming: {}'.format(poly_true_pos_streaming,poly_false_pos_streaming))
# print('Poly Precision Streaming: {:.4f}'.format(poly_true_pos_streaming/(poly_true_pos_streaming+poly_false_pos_streaming)*100))
# print('\nPoly True Positives Rat: {} | False Positives Rat: {}'.format(poly_true_pos_rat,poly_false_pos_rat))
# print('Poly Precision Rat: {:.4f}'.format(poly_true_pos_rat/(poly_true_pos_rat+poly_false_pos_rat)*100))



## -- 12 -- ##
from sklearn.neural_network import MLPClassifier
print('\n-- Classification based on Neural Networks --')
i5train=np.vstack((trainFeatures_files,trainFeatures_browsing,trainFeatures_images,trainFeatures_streaming, trainFeatures_rat))
i5Ctest=np.vstack((testFeatures_files,testFeatures_browsing,testFeatures_images,testFeatures_streaming, testFeatures_rat))

scaler = MaxAbsScaler().fit(i5train)
i5trainN=scaler.transform(i5train)
i5CtesttN=scaler.transform(i5Ctest)


alpha=1
max_iter=100000
clf = MLPClassifier(solver='lbfgs',alpha=alpha,hidden_layer_sizes=(20,),max_iter=max_iter)
clf.fit(i5trainN, o5trainClass) 
LT=clf.predict(i5CtesttN) 

ml_true_pos = 0
ml_false_pos = 0

ml_true_pos_files = 0
ml_false_pos_files = 0
ml_true_pos_browsing = 0
ml_false_pos_browsing = 0
ml_true_pos_images = 0
ml_false_pos_images = 0
ml_true_pos_streaming = 0
ml_false_pos_streaming = 0
ml_true_pos_rat = 0
ml_false_pos_rat = 0

nObsTest,nFea=i5CtesttN.shape
for i in range(nObsTest):
    if (LT[i]==o5testClass[i][0]):
        ml_true_pos+=1
    else:
        ml_false_pos+=1

    #detailed
    if (LT[i]==o5testClass[i][0] and o5testClass[i][0] == 0):
        ml_true_pos_files+=1
    elif (LT[i]!=o5testClass[i][0] and o5testClass[i][0] == 0):
        ml_false_pos_files+=1
    if (LT[i]==o5testClass[i][0] and o5testClass[i][0] == 1):
        ml_true_pos_browsing+=1
    elif (LT[i]!=o5testClass[i][0] and o5testClass[i][0] == 1):
        ml_false_pos_browsing+=1
    if (LT[i]==o5testClass[i][0] and o5testClass[i][0] == 2):
        ml_true_pos_images+=1
    elif (LT[i]!=o5testClass[i][0] and o5testClass[i][0] == 2):
        ml_false_pos_images+=1
    if (LT[i]==o5testClass[i][0] and o5testClass[i][0] == 3):
        ml_true_pos_streaming+=1
    elif (LT[i]!=o5testClass[i][0] and o5testClass[i][0] == 3):
        ml_false_pos_streaming+=1
    if (LT[i]==o5testClass[i][0] and o5testClass[i][0] == 4):
        ml_true_pos_rat+=1
    elif (LT[i]!=o5testClass[i][0] and o5testClass[i][0] == 4):
        ml_false_pos_rat+=1



    print('Obs: {:2} ({:<8}): Classification->{}'.format(i,Classes[o5testClass[i][0]],Classes[LT[i]]))

print('\nML True Positives: {} | False Positives: {}'.format(ml_true_pos,ml_false_pos))
print('ML Precision: {:.4f}'.format(ml_true_pos/(ml_true_pos+ml_false_pos)*100))

print('\nDetailed')

print('\nML True Positives Files: {} | False Positives Files: {}'.format(ml_true_pos_files,ml_false_pos_files))
print('ML Precision Files: {:.4f}'.format(ml_true_pos_files/(ml_true_pos_files+ml_false_pos_files)*100))
print('\nML True Positives Browsing: {} | False Positives Browsing: {}'.format(ml_true_pos_browsing,ml_false_pos_browsing))
print('ML Precision Browsing: {:.4f}'.format(ml_true_pos_browsing/(ml_true_pos_browsing+ml_false_pos_browsing)*100))
print('\nML True Positives Images: {} | False Positives Images: {}'.format(ml_true_pos_images,ml_false_pos_images))
print('ML Precision Images: {:.4f}'.format(ml_true_pos_images/(ml_true_pos_images+ml_false_pos_images)*100))
print('\nML True Positives Streaming: {} | False Positives Streaming: {}'.format(ml_true_pos_streaming,ml_false_pos_streaming))
print('ML Precision Streaming: {:.4f}'.format(ml_true_pos_streaming/(ml_true_pos_streaming+ml_false_pos_streaming)*100))
print('\nML True Positives Rat: {} | False Positives Rat: {}'.format(ml_true_pos_rat,ml_false_pos_rat))
print('ML Precision Rat: {:.4f}'.format(ml_true_pos_rat/(ml_true_pos_rat+ml_false_pos_rat)*100))