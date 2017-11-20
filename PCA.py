import numpy as np
import matplotlib.pyplot as plt

##features includes all n article
##
features=np.array(([1,9,0],[0,2,1]))#test value, modified later
##

def PCA(features,k):
    #k:number of principal components
    numfeat=features.shape[1]
    numarticle=features.shape[0]
    Cov=np.cov(features.T) ##take the covariance matrix for the features
    eval,evec=np.linalg.eig(Cov)

    ##find the first k principal components, where eval[k] is closest enough to zero
    #plt.plot(eval)
    #plt.show()


    evec_k=evec[0:k,:]
    mu=np.mean(features,axis=0)
    feat_new=np.zeros([numarticle,k])
    for i in range(numarticle):
        feat_new[i,:]=np.dot(evec_k,(feat_new[i,:]-mu))

    #the new features matrix is called feat_new

    return feat_new






