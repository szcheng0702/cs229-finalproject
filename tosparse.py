import numpy as np
import scipy.sparse

#output is a list of lists of tuples
#The largest list is called features

##
features=[[(0,1),(1,9)],[(1,2),(2,1)]]
dictsize=3 #size of the dictionary, i.e.how many words in total
##
#These are test data, will be replaced by the actual data later

def toSparse(features, dictsize):

    for i in range(len(features)):#i is the index of the article
        numwords=len(features[i])
        currenti_id=i*np.ones(numwords)
        currentj_id = np.array([word[0] for word in features[i]])
        currentdata=np.array([word[1] for word in features[i]])
        if i==0:
            i_id=currenti_id
            j_id=currentj_id
            data=currentdata
        else:
            i_id=np.append(i_id,currenti_id)
            j_id=np.append(j_id,currentj_id)
            data=np.append(data,currentdata)

    sp_features=scipy.sparse.coo_matrix((data,(i_id,j_id)),shape=(len(features),dictsize)).todense()

    return sp_features