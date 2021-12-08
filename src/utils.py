import numpy as np
from sklearn.utils import shuffle

 
def prepareInput(data, groupSize=int(8e4)):
    data = inferWeight(data)
    groups, weights = createGroupsRandom(data, groupSize)
    return groups, weights

# alpha and beta are knowledge in some framework, i.e. RLL.
# we don't include them in NeuCrowd
def inferWeight(data, alpha=None, beta=None):    
    votes = data[:, 1]
    maxVote = max(votes)
    weights = []
    for i in range(votes.shape[0]):
        v = votes[i]

        if v >= (1+maxVote)/2:
            if alpha is None or beta is None:
                weights.append(float(v/maxVote))
            else:
                weights.append(float((v+alpha)/(maxVote+alpha+beta)))
        else:
            if alpha is None or beta is None:
                weights.append(1-float(v/maxVote))
            else:
                weights.append(float((maxVote-v+alpha)/(maxVote+alpha+beta)))
    data[:, 1] = weights
    return data


def splitFeatureWeight(x):
    return x[:,1:], x[:,0]


def createGroupsRandom(data, groupSize=int(8e4)):
    positive = data[np.where(data[:, 0] == 1)]
    negative = data[np.where(data[:, 0] == 0)]
    posNum = positive.shape[0]
    negNum = negative.shape[0]

    idx = np.random.randint(low=0, high=posNum, size=groupSize)
    query = np.array([positive[i, 1:] for i in idx])
    posDoc = shuffle(query)

    idx = np.random.randint(low=0, high=negNum, size=groupSize)
    negDoc0 = np.array([negative[i, 1:] for i in idx])
    negDoc1 = shuffle(negDoc0)
    negDoc2 = shuffle(negDoc0)
    
    query, queryDocW = splitFeatureWeight(query)
    posDoc, posDocW = splitFeatureWeight(posDoc)
    negDoc0, negDoc0W = splitFeatureWeight(negDoc0)
    negDoc1, negDoc1W = splitFeatureWeight(negDoc1)
    negDoc2, negDoc2W = splitFeatureWeight(negDoc2)
    
    groups = (query, posDoc, negDoc0, negDoc1, negDoc2)
    weights = (queryDocW, posDocW, negDoc0W, negDoc1W, negDoc2W)
    return groups, weights
