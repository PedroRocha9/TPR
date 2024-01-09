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

########### Main Code #############
Classes={0:'Files',1:'Browsing',2:'Images', 3:'Streaming', 4:'Rat'}
plt.ion()
nfig=1

## -- 2 -- ##
features_normal=np.loadtxt("./45min_good-UA_mixed0.1AllF.dat")
features_mixed = np.loadtxt("./45min_all_mixed0.1AllF.dat")

oClass_normal=np.ones((len(features_normal),1))*0
oClass_mixed=np.ones((len(features_mixed),1))*1


features=np.vstack((features_normal,features_mixed))
oClass=np.vstack((oClass_normal,oClass_mixed))

# Assuming features and oClass are already defined
scalar = StandardScaler()
features_scaled = scalar.fit_transform(features)

pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_scaled)

plt.figure(figsize=(8,6))

# Define class labels
class_labels = ['Normal','Rat']
colors = ['purple', 'green']  # Define a color for each class

# Scatter plot
for i, label in enumerate(class_labels):
    plt.scatter(features_pca[oClass[:,0] == i, 0], features_pca[oClass[:,0] == i, 1], 
                label=label, color=colors[i])

# plt.xlabel('First Principal Component')
# plt.ylabel('Second Principal Component')

# Add a legend
plt.legend(title="Classes")

plt.show(block=True)