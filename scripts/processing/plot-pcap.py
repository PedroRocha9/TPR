import numpy as np
import matplotlib.pyplot as plt
import sys

filename = sys.argv[1]

datFile = filename
data = np.loadtxt(datFile,dtype=int)

#show two plots at the same time

plt.plot(data[:,1],'k')
plt.title("Upload Bytes")
plt.figure()
plt.plot(data[:,3],'b')
plt.title("Download Bytes")
plt.show()


# plt.plot(data[:,1],'k')
# plt.show()
# plt.plot(data[:,3],'b')
# plt.show()