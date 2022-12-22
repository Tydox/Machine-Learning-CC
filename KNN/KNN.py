#%% md
# Full implemetation of KNN in Python
#%% md
#### Instructions
#%% md
Complete this notebook and submit it as the solution for question 2 in home assignment no. 1 

* In the relevant cell below, fill in the two id numbers for the students submitting this notebook
* Cells with 'COMPLETED - NO CHANGES ARE REQUIRED' in their description require no edits and should be left unchanged
* Cells with 'TO BE COMPLETED' require you to complete the code according to the given description 
* **NOTE**: After you complete the notebook and run all cells, **discuss** the results displayed in the graph produced at the end of the notebook 
#%%
# replace the strings below with the personal details of the two students submitting this notebook
Name1 = 'Danno' # student 1
Name2 = 'Omeriko' # student 2
id1 = '123' # student 1 id
id2 = '456' # student 2 id
#%%
import numpy as np

def LoadIrisData(fname):
    '''
    ***COMPLETED - NO CHANGES ARE REQUIRED***
    
    Usage: load data from file in the path 'fname'
    
    Input: the path for the file that stores the data
    Output: feature matrix X, and labels vector Y
    '''
    
    # read the features data from the csv file
    X = np.loadtxt(fname,dtype=float, delimiter=',', skiprows = 1,usecols=[1,2,3,4])

    # read the labels data from the csv file
    Y = np.loadtxt(fname,dtype=str, delimiter=',', skiprows = 1,usecols=[5])

    inds_class1 = np.where(Y=='setosa')[0]
    inds_class2 = np.where(Y=='virginica')[0]
    inds_class3 = np.where(Y=='versicolor')[0]

    plt.scatter(X[Y=='setosa',2],X[Y=='setosa',1],label='setosa')
    plt.scatter(X[Y=='virginica',2],X[Y=='virginica',1],label='virginica')
    plt.scatter(X[Y=='versicolor',2],X[Y=='versicolor',1],label='versicolor')
    plt.legend()
    plt.show()



    return X, Y
#%%
from numpy.random import randint
def SplitTrainTest(X,Y):
    '''
    ***COMPLETED - NO CHANGES ARE REQUIRED***
    
    Input: feature matrix X, labels vector Y
    Output: X_train, Y_train - train data
            X_test, Y_test - test data
    '''
    
    # permute the ordering of the examples
    ind = np.random.RandomState(seed=int(id1)+int(id2)).permutation(len(Y))

    # randomly choose the size of the training data
    Ntrain = np.random.RandomState(seed=int(id1)+int(id2)).randint(0.6*len(Y),0.8*len(Y))

    # split the data into train and test datasets
    X_train = X[ind[:Ntrain]]
    Y_train = Y[ind[:Ntrain]]
    X_test  = X[ind[Ntrain:]]
    Y_test  = Y[ind[Ntrain:]]
    
    return X_train, Y_train, X_test, Y_test
#%%
def PairwiseDistance(a,b):
    '''
    ***TO BE COMPLETED***
    
    Usage: compute the euclidean distance between pair of examples
    
    Input: two numerical vectors of eqaul length
    Output: the euclidean distance between the vectors
    
    Example: 
    PairwiseDistance([3, 1],[6, 5]) returns 5
    [calculation: sqrt((3-6)^2+(1-5)^2) = sqrt(9+16) = 5]
    '''
    
    
    return np.sqrt(np.sum((a-b)**2))
#%%
def SortArray(a):
    '''
    ***TO BE COMPLETED***
    
    Usage: return the ordering of the elements of the input array such that their values are ascending
    
    Input: a vector with numerical values
    Output: the indices of the input array, such that the array is sorted in ascending order
    
    Example: 
    SortArray([1.34, 2.18, 1.21]) returns [2,0,1]
    '''
    
    return np.argsort(a)
#%%
def MajorityVote(inds,y,K):
    '''
    ***TO BE COMPLETED***
    
    Usage: find the most frequent label in y's K leading values, after it sorted acording to inds
    
    Input: inds - an ordering of y
           y    - a vector of labels
           K    - an integer
    Output: most frequent label
    
    Example 1: 
    MajorityVote(inds = [2, 0, 1, 3, 5, 4],
              y    = ['setosa', 'setosa', 'versicolor', 'setosa', 'virginica', 'versicolor']) 
              K    = 1)
    returns 'versicolor'

    Example 2: 
    MajorityVote(inds = [2, 0, 1, 3, 5, 4],
              y    = ['setosa', 'setosa', 'versicolor', 'setosa', 'virginica', 'versicolor']) 
              K    = 3)
    returns 'setosa'
    '''
    _indx = np.array(inds[:K])
    _sortArr = y[_indx]
    return max(set(list(_sortArr)), key = list(_sortArr).count)


    # the labels of the K nearest neighbors
    
   
    # counting the number occurences of each label amongst the K nearest neighbors
    
    
    # the most frequent label amongst the K nearest neighbors

#%%
def KNearestNeighborsClassifier(X_train, Y_train , X_test, K):
    '''
    ***TO BE COMPLETED***
    
    Usage: estimate the label of each unclassified example in X_test to be the most 
           frequent label amongst its K nearest neighbors in X_train
           
    Input:  X_train - training data (features)
            Y_train - training data (labels) 
            X_test  - test data (features)
            K       - number of neighbors
    
    Output: Y_test  - the classifier's labels for the test data
    
    '''
    Y_pred = []

    # loop through the examples to be classified
    for sample in X_test:
        # compute the pairwise distances between the each unclassified example and 
        # all classified example
        _dstVec  = []
        for train in X_train:

            _dst = PairwiseDistance(train,sample)
            _dstVec.append(_dst)
        
        # find the unclassified example's K nearest neighbors
        _idxs = SortArray(_dstVec)
        estLabel = MajorityVote(_idxs,Y_train,K)
        Y_pred.append(estLabel)
    return Y_pred
#%%
import matplotlib.pyplot as plt

def PlotAccuracy(accuracy):
    '''
    ***COMPLETED - NO CHANGES ARE REQUIRED***
    '''
    plt.figure(figsize=(14,5))
    plt.plot(accuracy,'.-')
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.title('KNN estimated accuracy for various values of K')
    plt.xlim([0, 100])
    plt.ylim([0, 1])
    return
#%%
def Accuracy(Y_pred, Y_test):
    '''
    ***TO BE COMPLETED***
    
    Usage: calculate the classifier accuracy 
    
    Input: Y_test - true labels of X_test
           Y_pred - estimated labels of X_test
    Output: accuracy - the nember of correctly classified examples, divided by the total number of examples    

    Example: 
    calling Accuracy(Y_pred = ['setosa', 'setosa', 'versicolor', 'setosa', 'virginica']), 
                     Y_test = ['setosa', 'setosa', 'versicolor', 'versicolor', 'virginica']))
    returns 0.8
    '''
    
    # calculate the accuracy - the ratio between the number of examples in X_test for which
    # the true label and the estimated label are the same, and the total number 
    # of examples in X_test

    return np.sum(Y_pred==Y_test)/(len(Y_pred))
#%%

def main(fname, Kmax):

    # STEP 1: load data
    X,Y = LoadIrisData(fname)
    
    # STEP 2: split the data into train/test datasets
    X_train, Y_train, X_test, Y_test = SplitTrainTest(X,Y)
    print('Data is split into ' + str(X_train.shape[0]) + ' examples for training and ' + str(X_test.shape[0]) + ' examples for testing')
    
    # an array to store all computed accuracies
    accuracy = np.zeros(Kmax)
    
    # repeat for all considered values of K
    for K in range(Kmax):
        # STEP 3: classify the test data using a KNN classifier
        Y_pred = KNearestNeighborsClassifier(X_train, Y_train, X_test , K+1)

        # STEP 4: calculate the KNN classifier accuracy
        accuracy[K] = Accuracy(Y_pred, Y_test)
        
    # plot results
    PlotAccuracy(accuracy)
    return

#%%
'''
fname = 'iris.csv'
Kmax = 100
main(fname, Kmax)
'''
import time
# get the start time
st = time.time()

fname = 'iris.csv'
Kmax = 100
main(fname, Kmax)

    # get the execution time
elapsed_time = time.time() - st
print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
#%%
# replace the strings below with the personal details of the two students submitting this notebook
Name1 = 'Daniel Mizrahi' # student 1 name and surname in the format 'Arik Paran'
Name2 = 'Omer Ozeri' # student 2 name and surname in the format 'Arik Paran'
id1 = '208544072' # student 1 id
id2 = '313512550' # student 2 id

import numpy as np

def LoadIrisData(fname):
    '''
    ***COMPLETED - NO CHANGES ARE REQUIRED***

    Usage: load data from file in the path 'fname'

    Input: the path for the file that stores the data
    Output: feature matrix X, and labels vector Y
    '''

    # read the features data from the csv file
    X = np.loadtxt(fname,dtype=float, delimiter=',', skiprows = 1,usecols=[1,2,3,4])

    # read the labels data from the csv file
    Y = np.loadtxt(fname,dtype=str, delimiter=',', skiprows = 1,usecols=[5])

    inds_class1 = np.where(Y=='setosa')[0]
    inds_class2 = np.where(Y=='virginica')[0]
    inds_class3 = np.where(Y=='versicolor')[0]

    plt.scatter(X[Y=='setosa',2],X[Y=='setosa',1],label='setosa')
    plt.scatter(X[Y=='virginica',2],X[Y=='virginica',1],label='virginica')
    plt.scatter(X[Y=='versicolor',2],X[Y=='versicolor',1],label='versicolor')
    plt.legend()
    plt.show()



    return X, Y

from numpy.random import randint
def SplitTrainTest(X,Y):
    '''
    ***COMPLETED - NO CHANGES ARE REQUIRED***

    Input: feature matrix X, labels vector Y
    Output: X_train, Y_train - train data
            X_test, Y_test - test data
    '''

    # permute the ordering of the examples
    ind = np.random.RandomState(seed=randint(0,100000)).permutation(len(Y))

    # randomly choose the size of the training data
    Ntrain = np.random.RandomState(seed=randint(0,100000)).randint(0.6*len(Y),0.8*len(Y))

    # split the data into train and test datasets
    X_train = X[ind[:Ntrain]]
    Y_train = Y[ind[:Ntrain]]
    X_test  = X[ind[Ntrain:]]
    Y_test  = Y[ind[Ntrain:]]

    return X_train, Y_train, X_test, Y_test


def PairwiseDistance(a,b):
    '''
    ***TO BE COMPLETED***

    Usage: compute the euclidean distance between pair of examples

    Input: two numerical vectors of eqaul length
    Output: the euclidean distance between the vectors

    Example:
    PairwiseDistance([3, 1],[6, 5]) returns 5
    [calculation: sqrt((3-6)^2+(1-5)^2) = sqrt(9+16) = 5]
    '''


    return np.sqrt(np.sum((a-b)**2))



def SortArray(a):
    '''
    ***TO BE COMPLETED***

    Usage: return the ordering of the elements of the input array such that their values are ascending

    Input: a vector with numerical values
    Output: the indices of the input array, such that the array is sorted in ascending order

    Example:
    SortArray([1.34, 2.18, 1.21]) returns [2,0,1]
    '''

    return np.argsort(a)



def MajorityVote(inds,y,K):
    '''
    ***TO BE COMPLETED***

    Usage: find the most frequent label in y's K leading values, after it sorted acording to inds

    Input: inds - an ordering of y
           y    - a vector of labels
           K    - an integer
    Output: most frequent label

    Example 1:
    MajorityVote(inds = [2, 0, 1, 3, 5, 4],
              y    = ['setosa', 'setosa', 'versicolor', 'setosa', 'virginica', 'versicolor'])
              K    = 1)
    returns 'versicolor'

    Example 2:
    MajorityVote(inds = [2, 0, 1, 3, 5, 4],
              y    = ['setosa', 'setosa', 'versicolor', 'setosa', 'virginica', 'versicolor'])
              K    = 3)
    returns 'setosa'
    '''
    # the labels of the K nearest neighbors


    # counting the number occurences of each label amongst the K nearest neighbors


    # the most frequent label amongst the K nearest neighbors
    _indx = np.array(inds[:K])
    _sortArr = y[_indx]
    return max(set(list(_sortArr)), key = list(_sortArr).count)








def KNearestNeighborsClassifier(X_train, Y_train , X_test, K):
    '''
    ***TO BE COMPLETED***

    Usage: estimate the label of each unclassified example in X_test to be the most
           frequent label amongst its K nearest neighbors in X_train

    Input:  X_train - training data (features)
            Y_train - training data (labels)
            X_test  - test data (features)
            K       - number of neighbors

    Output: Y_test  - the classifier's labels for the test data

    '''
    Y_pred = []

    # loop through the examples to be classified
    for sample in X_test:
        # compute the pairwise distances between the each unclassified example and
        # all classified example
        _dstVec  = []
        for train in X_train:

            _dst = PairwiseDistance(train,sample)
            _dstVec.append(_dst)

        # find the unclassified example's K nearest neighbors
        _idxs = SortArray(_dstVec)
        estLabel = MajorityVote(_idxs,Y_train,K)
        Y_pred.append(estLabel)
    return Y_pred



import matplotlib.pyplot as plt

def PlotAccuracy(accuracy):
    '''
    ***COMPLETED - NO CHANGES ARE REQUIRED***
    '''
    plt.figure(figsize=(14,5))
    plt.plot(accuracy,'.-')
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.title('KNN estimated accuracy for various values of K')
    plt.xlim([0, 100])
    plt.ylim([0, 1])
    return




def Accuracy(Y_pred, Y_test):
    '''
    ***TO BE COMPLETED***

    Usage: calculate the classifier accuracy

    Input: Y_test - true labels of X_test
           Y_pred - estimated labels of X_test
    Output: accuracy - the nember of correctly classified examples, divided by the total number of examples

    Example:
    calling Accuracy(Y_pred = ['setosa', 'setosa', 'versicolor', 'setosa', 'virginica']),
                     Y_test = ['setosa', 'setosa', 'versicolor', 'versicolor', 'virginica']))
    returns 0.8
    '''

    # calculate the accuracy - the ratio between the number of examples in X_test for which
    # the true label and the estimated label are the same, and the total number
    # of examples in X_test

    return np.sum(Y_pred==Y_test)/(len(Y_pred))




'''
def main(fname, Kmax):

    # STEP 1: load data
    X,Y = LoadIrisData(fname)

    # STEP 2: split the data into train/test datasets
    X_train, Y_train, X_test, Y_test = SplitTrainTest(X,Y)
    print('Data is split into ' + str(X_train.shape[0]) + ' examples for training and ' + str(X_test.shape[0]) + ' examples for testing')

    # an array to store all computed accuracies
    accuracy = np.zeros(Kmax)

    # repeat for all considered values of K
    for K in range(Kmax):
        # STEP 3: classify the test data using a KNN classifier
        Y_pred = KNearestNeighborsClassifier(X_train, Y_train, X_test , K+1)

        # STEP 4: calculate the KNN classifier accuracy
        accuracy[K] = Accuracy(Y_pred, Y_test)

    # plot results
    PlotAccuracy(accuracy)
    return
'''

def main(fname, Kmax):
    '''
    ***COMPLETED - NO CHANGES ARE REQUIRED***
    '''

    # STEP 1: load data
    X,Y = LoadIrisData(fname)
    endSize = 100

    mat = np.zeros([100,100])


    for i in range(0,endSize):
        X_train, Y_train, X_test, Y_test = SplitTrainTest(X,Y)
        #print(X_train.shape)
        #print(X_test.shape)
        #print('Data [' + str(i) + '] is split into ' + str(X_train.shape[0]) + ' examples for training and ' + str(X_test.shape[0]) + ' examples for testing')
        accuracy = np.zeros(Kmax)

        for K in range(Kmax):
            Y_pred = KNearestNeighborsClassifier(X_train, Y_train, X_test , K+1)
            accuracy[K] = Accuracy(Y_pred, Y_test)

        #PlotAccuracy(accuracy)
        mat[i] = accuracy
    if(endSize>1):
        newAcc = np.zeros([1,endSize]).transpose()

        for i in range(0,endSize):
            #print(f"Matix: {mat[:,i]} \n")
            avg = sum(mat[:,i] ) / endSize
            print(f"Avg [{i}] =\t{avg}")
            newAcc[i] = avg

        PlotAccuracy(newAcc)


    return




'''
fname = 'iris.csv'
Kmax = 100
main(fname, Kmax)
'''
import time
# get the start time
st = time.time()

fname = 'iris.csv'
Kmax = 100
main(fname, Kmax)

    # get the execution time
elapsed_time = time.time() - st
print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
#%%
