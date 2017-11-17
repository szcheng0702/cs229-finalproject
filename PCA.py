import numpy as np
import matplotlib.pyplot as plt

##features includes all n article
##
features=np.array(([1,9,0],[0,2,1]))#test value, modified later
##

numfeat=features.shape[1]
numarticle=features.shape[0]
Cov=np.cov(features)
eval,evec=np.linalg.eig(Cov)

##find the first k principal components, where eval[k] is closest enough to zero
#plt.plot(eval)
#plt.show()

##
k=2 #num of Principal components, modified later
##

evec_k=evec[1:k,:]
mu=np.mean(features,axis=1)
feat_new=np.zeros([numarticle,k])
for i in range(numfeat):
    feat_new[i,:]=evec_k.T*(evec_k*(features[:,i]-mu))

#the new features matrix is called feat_new





