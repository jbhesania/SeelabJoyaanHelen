'''
This model is meant to allow bitshifting by converting HV to powers of 2. It is a WIP.
'''

# coding: utf-8

# In[17]:

import numpy as np
import random
import pickle
from numpy import linalg as li
import time
import scipy
import scipy.cluster
import sklearn.preprocessing
from scipy.special import xlogy
import sys


# In[18]:

# genRandomHV(): generates random hypervector
# D: number of dimensions
def genRandomHV(D):
    if (D % 2) == 1:
        print("Dimension is odd.")
    else:
        hv = np.random.permutation(D) #create a sequence with random values from 0 to D
        for x in range(D):
            if hv[x] < D/2:
                hv[x] = -1
            else:
                hv[x] = 1
        return hv

#generate D feature hypervectors and return them as an array
def genFeatureHV(D):
    arr = np.empty([D, 10000]) #2D array of D rows
    for i in range(0, D):
        arr[i] = genRandomHV(10000)
    return arr

# creates G levels between a and b; level hypervectors
# z is number of bits to flip
def genLevelHV(g, z):
    arr = np.empty([g, 10000])
    arr[0] = genRandomHV(10000)
    for i in range(1, g):
        arr[i] = arr[i-1]
        for j in range (0, z):
            a = int(random.uniform(0, 1) * 10000)  #chooses a random index and switches value 
            if arr[i, a] is -1:
                arr[i, a] = 1
            else:
                arr[i, a] = -1
    return arr.astype(int)


#create D empty hypervectors filled to 0 for classHV
def genClassHV(D, numSubClasses):
    arr = np.zeros([D, numSubClasses, 10000])
    return arr

def findMinMax(A):
    return int(round(max(A))), int(round(min(A)))

def createBins(BIN_NUM, arr, i):
    maxValue, minValue = findMinMax(arr[:, i])
    bins = []
    for j in range(BIN_NUM):
            bins = np.append(bins, minValue + (float(maxValue-minValue)/BIN_NUM)*j)
    return bins


# In[30]:

#if guessed wrong, subtract from wrong hypervector, add to correct hypervector
#repeat until accepted accuracy
def retrain(classHV, testArr, levelHV, featureHV, numSubClasses):
    numCol = len(testArr[0,:])  
    for i in range (0, len(testArr)):
        queryHV = testArr[i, 0:(numCol - 1)]
        cosVals = cosineSimilarity(classHV, queryHV)
        x = findAccuracy(cosVals, testArr, numCol, i, numSubClasses)
        if x is 0:
            maxVal = int(np.argmax(cosVals))
            classNum = int(maxVal / numSubClasses)
            subClassNum = int(maxVal % numSubClasses)
            
            #delete queryHV from incorrect classHV
            #classHV[classNum][subClassNum] = classHV[classNum][subClassNum] - queryHV
            classHV[classNum][:] = classHV[classNum][:] - queryHV

            
            #add queryHV to correct classHV
            #find subclassHV with highest cosine similarity
            trueClass = int(testArr[i, numCol-1])
            trueSub = int(np.argmax(cosVals[trueClass]))
            classHV[trueClass, :] = classHV[trueClass, :] + queryHV
            
                 
    return classHV
    
def spliceData (fileName):
    f = open(fileName, "r")
    numFeatures = int(f.readline())
    numClasses = int(f.readline())

    array = np.empty([1, numFeatures + 1])
    array[0] =  f.readline().split(',')
    
    for line in f:
        feature1, feature2, classNum = line.split(',')
        data = np.empty([1,numFeatures + 1])
        data = [float(feature1), float(feature2), int(classNum)]
        array = np.vstack((array, data)) #adds a new row to array
    f.close()
    
    #array[:, 0:(len(array[0])-1)] = sklearn.preprocessing.normalize(array[:, 0:(len(array[0])-1)], axis = 1)
    
    print(array) 
    
    return numFeatures, numClasses, array
    
#finds cosine similarity between the queryHV and each classHV
def cosineSimilarity (classHV, queryHV):
    cosVals = np.zeros([len(classHV), len(classHV[0])])
    for i in range(0, len(classHV)):
        for j in range(0, len(classHV[0])):
            #cosine similarity
            cosVals[i][j] = np.dot(queryHV, classHV[i][j])/ (np.linalg.norm(queryHV)*np.linalg.norm(classHV[i][j])) 
    return cosVals

#checks if the classHV with the highest cosine similarity is same class as the actual class of the queryHV
def findAccuracy(cosVals, testOverallArr, testArrNumCols, i, numSubClasses):
    #np.argmax returns value from array as if its a 1 x n array (if is 2x2 array, will return numbers 1,2,3,4, and the like)
    #since classes have same number of subclasses, then dividing by numSubClasses should get which class its in, and modulus
    #should get which subclass it is
    maxVal = int(np.argmax(cosVals))
    classNum = int(maxVal / numSubClasses)
    
    #TODO
    #print(classNum, int(testOverallArr[i,testArrNumCols - 1]))
    if classNum == int(testOverallArr[i,testArrNumCols - 1]):
        return 1
    else:
        return 0


# In[20]:

#uses kmeans method on given array, which will all be of the same class
#adds hypervectors to destined class hypervector
#returns classHV
def kmeans(array, numSubClasses):
    #kmeans only supports value that are doubles or floats
    array = array.astype(float)
    #'random', 'points', 'uniform'
    centroid, label = scipy.cluster.vq.kmeans2(array, numSubClasses, minit='points')
    return label


# In[21]:

def unpickle(fileName):
    with open(fileName, 'rb') as handle:
        featureArr = pickle.load(handle)
        featureArr = np.asarray(featureArr)
        numFeatures = len(featureArr[0,:])
        classArr = pickle.load(handle)
        classArr = np.asarray(classArr)
        numClasses = len(set(classArr))
        arr = np.c_[featureArr, classArr]
        
    return numFeatures, numClasses, arr

#Change all values in array to powers of 2
def changePowers(array):
  # Save signs and then perform masked log 
  sign = np.sign(array)
  arrayN = np.absolute(array)

  # Fill masked places with 0 and divide by log2 (log math) 
  arrayN = np.ma.log(arrayN)
  arrayN = arrayN.filled(0) 
  arrayN = arrayN / np.log(2)

  arrayN = np.trunc(arrayN) 

  # Raise 2 to the power of arrayN (2^n) and redistribute signs
  arrayof2 = np.full(arrayN.shape, 2)
  arrayN = np.power(arrayof2, arrayN)
  arrayN = arrayN*sign


  # Repeat for the difference in array and arrayN
  array = array - arrayN
  sign = np.sign(array)
  array = np.absolute(array)
  
  # Masked log and refill
  array = np.ma.log(array)
  array = array.filled(0)
  array = array / np.log(2)
  
  array = np.round(array)

  # Raise 2 to the power of arrayN (2^n) and redistribute signs
  array = np.power(arrayof2, array)
  array = array*sign

  array = arrayN + array
  return array

#############################################################################
######################  END OF METHODS #####################################
############################################################################


#trainingFile = 'dataset/IRISPickles/iris_train.pickle'
#testingFile = 'dataset/IRISPickles/iris_test.pickle'
#trainingFile = 'dataset/ISOLETPickles/ISOLET_train.pickle'
#testingFile = 'dataset/ISOLETPickles/ISOLET_test.pickle'
#trainingFile = 'dataset/PAMPA2Pickles/PAMPA2_train.pickle'
#testingFile = 'dataset/PAMPA2Pickles/PAMPA2_test.pickle'
#trainingFile = 'dataset/UCIHARPickles/sa_train.pickle'
#testingFile = 'dataset/UCIHARPickles/sa_test.pickle'
#trainingFile = 'dataset/moons/moons_2048_train.txt'
#testingFile = 'dataset/moons/moons_2048_test.txt'
#trainingFile = "dataset/blob_train.txt"
#testingFile = "dataset/blob_test.txt"
trainingFile = "dataset/FACEPickles/face_train.pickle"
testingFile = "dataset/FACEPickles/face_test.pickle"


#number of features, classes, and array of all values
F, C, overallArr = unpickle(trainingFile)

numSubClasses = 64
#int(sys.argv[1])
BIN_NUM = 20
FLIP_NUM = 50
numValidation = 50
#int(sys.argv[2])


#the feature hypervector
featureHV = genFeatureHV(F)
classHV = genClassHV(C, numSubClasses)

arrNumRows = len(overallArr)
arrNumCols = len(overallArr[0,:])

print "Successful unpickle with ", numSubClasses, " subclasses and ", numValidation, " retraining."


# In[24]:

#READING THE DATA

# the process of changing from values (floats) to ints that represent what bin theyre in
levelHV = np.zeros([F, BIN_NUM, 10000])
for i in range(0, arrNumCols - 1):
    bins = createBins(BIN_NUM, overallArr, i)
    #test for different range values that cause column sizes to be different
    levelHV[i] = genLevelHV(BIN_NUM, FLIP_NUM)
    feature1 = overallArr[:arrNumRows, i]
    overallArr[:arrNumRows, i] = np.digitize(feature1,bins) - 1 
overallArr = overallArr.astype(int) 
print("Done encoding")


# In[25]:

#CREATING THE CLASS HVs
for i in range(0, len(classHV)):
    print(i)
    #grabs all HVs that are of a specific class
    arr = overallArr[overallArr[:, arrNumCols - 1] == i]
    numCols = len(arr[0,:])
    
    #grabs list of values saying which HV is to which subclass
    label = kmeans(arr, numSubClasses)
    
    #creates subclass hypervectors for that specific class
    for j in range(0, len(arr)):
        for k in range(0, numCols - 1):
            #arr[j,k] grabs which bin the HV is in, since the array was digitized earlier    
            levelOne = levelHV[k, arr[j, k]]
            productOne = levelOne * featureHV[k]
            
            #classHV[class, subclass]
            classHV[i, label[j]] = classHV[i, label[j]] + productOne 


classHV = changePowers(classHV)
print("Done creating classHV")


# In[26]:

################## reading the testing data ####################

# starting to test data, get data into overalltestarray, with fourth column of 0 for false and 1 for true
#number of features, classes, and array of all values
testF, testC, testOverallArr = unpickle(testingFile)
    
testArrNumRows = len(testOverallArr)
testArrNumCols = len(testOverallArr[0,:])


# the process of changing from values (floats) to ints that represent what bin theyre in
for i in range(0, testArrNumCols - 1):
    feature1 = testOverallArr[:testArrNumRows, i]
    testOverallArr[:testArrNumRows, i] = np.digitize(feature1,bins) - 1 #returns value of bin feature1 belongs to

testOverallArr = testOverallArr.astype(int)

print("Done reading test data")


# In[27]:

#Creates all query HV for training dataset and testing dataset

trainQueryHV = np.empty([len(overallArr), 10000])
for i in range(0, len(overallArr)):
  queryHV = np.zeros([10000])
  for j in range(0, arrNumCols - 1):
    levelOne = levelHV[j, overallArr[i, j]]
    productOne = levelOne * featureHV[j]
    queryHV = queryHV + productOne
  
  trainQueryHV[i] = queryHV


a = overallArr[:, arrNumCols - 1].reshape(-1, 1)
print(a.shape)
trainQueryHV = changePowers(trainQueryHV)

trainQueryHV = np.c_[trainQueryHV, a]

testQueryHV = np.zeros([testArrNumRows, 10000])
# encodes each test HV
for i in range(0, testArrNumRows):
  queryHV = np.zeros([10000])
  for j in range(0, testArrNumCols - 1):
    levelOne = levelHV[j, testOverallArr[i, j]]
    productOne = levelOne * featureHV[j]
    queryHV = queryHV + productOne
    
  testQueryHV[i] = queryHV

testQueryHV = changePowers(testQueryHV)

print("Done creating encoded train/test HV")


# In[28]:


# In[ ]:

Y = np.zeros([numValidation])
print("Retrainings out of " + str(numValidation))
for k in range(numValidation):
    
    accuracy = np.zeros([testArrNumRows])
    
    #calculate the cosine similarity for each of the class HVs
    for i in range(0, len(testQueryHV)):
      cosVals = cosineSimilarity(classHV, testQueryHV[i])
        #if the smallest cosines class is the actual class
      accuracy[i] = findAccuracy(cosVals, testOverallArr, testArrNumCols, i, numSubClasses)
        
    Y[k] = np.sum(accuracy)/float(len(testOverallArr))
   
    classHV = retrain(classHV, trainQueryHV, levelHV, featureHV, numSubClasses) 
    classHV = changePowers(classHV)
    #classHV = (classHV +  retrain(classHV, trainQueryHV, levelHV, featureHV, numSubClasses))/2

    print(k)



# In[ ]:

print(np.sum(accuracy), len(accuracy))
print(np.sum(accuracy)/len(testOverallArr))
print(len(testOverallArr))
print(Y)

# In[ ]:

#import matplotlib.pyplot as plt
#plt.plot(Y)
#plt.show()


# In[ ]:


#VALIDATION TESTING on moons

# 20 validation
# without subtraction + add to all classHV: 0.7587890625
# without subraction + only add to subclass: 0.74609375
# with subtraction + add to all class HV : 0.8720703125
# with subtraction from all + add to subclass: 0.7626953125
# ****** with subtraction from all + add to all: 0.90478515625 ******

# with subtraction from all + add to all
# 50 validation: 0.8818359375
# 100 validation: 0.90576171875



# AVERAGING on moons
# 5 validation:
# 20 validation: 0.90478515625
# 50 validation: 0.9033203125
# 100 validation: 0.8984375
# 200 validation: 














