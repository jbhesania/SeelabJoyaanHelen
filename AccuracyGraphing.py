'''
This file runs retraining normally but also gets an accuracy on the testing data after each retraining to give us an idea of the changes during retraining. 
It is not the correct computing file.
'''
import numpy as np
import random
import pickle
from numpy import linalg as li
import scipy
import scipy.cluster


# In[18]:

# genRandomHV(): generates random hypervector
# D: number of dimensions
def genRandomHV(D):
    if (D % 2) == 1:
        print("Dimension is odd.")
    else:
        hv = np.random.permutation(D) #create a sequence with random values from 0 to D
        for x in range(D): #all values in hypervector will be either -1 or 1
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

# creates g levels between a and b; creates a level hypervector
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

#finds the min and max value of the array A
def findMinMax(A):
    return int(round(max(A))), int(round(min(A)))

#creates bins; used when creating the class HVs
def createBins(BIN_NUM, arr, i):
    maxValue, minValue = findMinMax(arr[:, i])
    bins = []
    for j in range(BIN_NUM):
            bins = np.append(bins, minValue + (float(maxValue-minValue)/BIN_NUM)*j)
    return bins

#if guessed wrong, subtract from wrong hypervector, add to correct hypervector
def retrain(classHV, testArr, levelHV, featureHV, numSubClasses):
    numCol = len(testArr[0,:])  
    for i in range (0, len(testArr)):
        queryHV = testArr[i, 0:(numCol - 1)]
        cosVals = cosineSimilarity(classHV, queryHV)
        x = findAccuracy(cosVals, testArr, numCol, i, numSubClasses)
        if x is 0:
            #finds correct subclass HV
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
   
#reads in unpickled/regular files of a specific format
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
    
    if classNum == int(testOverallArr[i,testArrNumCols - 1]):
        return 1
    else:
        return 0

#uses kmeans method on given array, which will all be of the same class
#adds hypervectors to destined class hypervector
#returns classHV
def kmeans(array, numSubClasses):
    #kmeans only supports value that are doubles or floats
    array = array.astype(float)
    #'random', 'points', 'uniform'
    centroid, label = scipy.cluster.vq.kmeans2(array, numSubClasses, minit='points')
    return label

def unpickle(fileName):
    with open(fileName, 'rb') as handle:
        featureArr = pickle.load(handle)
        featureArr = np.array(featureArr)
        numFeatures = len(featureArr[0,:])
        classArr = pickle.load(handle)
        classArr = np.array(classArr)
        numClasses = len(set(classArr))
        arr = np.c_[featureArr, classArr]
    return numFeatures, numClasses, arr

###########################################################################
######################### END OF METHODS ##################################
###########################################################################

#all datasets used for training and testing

#trainingFile = 'dataset/ISOLETPickles/isolet_train.pickle'
#testingFile = 'dataset/ISOLETPickles/isolet_test.pickle'
#trainingFile = 'dataset/PAMPA2Pickles/PAMPA2_train.pickle'
#testingFile = 'dataset/PAMPA2Pickles/PAMPA2_test.pickle'
trainingFile = 'dataset/UCIHARPickles/sa_train.pickle'
testingFile = 'dataset/UCIHARPickles/sa_test.pickle'
#trainingFile = 'dataset/moons/moons_2048_train.txt'
#testingFile = 'dataset/moons/moons_2048_test.txt'
#trainingFile = "dataset/blob_train.txt"
#testingFile = "dataset/blob_test.txt"
#trainingFile = "dataset/FACEPickles/face_train.pickle"
#testingFile = "dataset/FACEPickles/face_test.pickle"

#number of features, classes, and array of all values
F, C, overallArr = unpickle(trainingFile)


numSubClasses = 1
BIN_NUM = 1000
FLIP_NUM = 50
numValidation = 50


#the feature hypervector
featureHV = genFeatureHV(F)

classHV = genClassHV(C, numSubClasses)

arrNumRows = len(overallArr)
arrNumCols = len(overallArr[0,:])

print "Successful unpickle of with ", trainingFile, numSubClasses, " subclasses and ", numValidation, " retraining."


#READING THE DATA

# the process of changing from values (floats) to ints that represent what bin theyre in
levelHV = np.zeros([F, BIN_NUM, 10000])
copyOverallArr = overallArr.copy()

for i in range(0, arrNumCols - 1):
    bins = createBins(BIN_NUM, overallArr, i)
    #test for different range values that cause column sizes to be different
    levelHV[i] = genLevelHV(BIN_NUM, FLIP_NUM)
    feature1 = overallArr[:arrNumRows, i]
    overallArr[:arrNumRows, i] = np.digitize(feature1,bins) - 1 
overallArr = overallArr.astype(int) 
print("Done encoding")

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
    
print("Done creating classHV")


################## reading the testing data ####################

# starting to test data, get data into overalltestarray, with fourth column of 0 for false and 1 for true
#number of features, classes, and array of all values
testF, testC, testOverallArr = unpickle(testingFile)
testOverallArr = testOverallArr[:int(len(testOverallArr)]
testArrNumRows = len(testOverallArr)
testArrNumCols = len(testOverallArr[0,:])

# the process of changing from values (floats) to ints that represent what bin theyre in
for i in range(0, testArrNumCols - 1):
    feature1 = testOverallArr[:testArrNumRows, i]
    testOverallArr[:testArrNumRows, i] = np.digitize(feature1,bins) - 1 #returns value of bin feature1 belongs to

testOverallArr = testOverallArr.astype(int)


print("Done reading test data from ", testingFile)


#Creates all query HV for training dataset and testing dataset

trainQueryHV = np.empty([len(overallArr), 10000])
for i in range(0, len(overallArr)):
  queryHV = np.zeros([10000])
  for j in range(0, arrNumCols - 1):
    levelOne = levelHV[j, overallArr[i, j]]
    productOne = levelOne * featureHV[j]
    queryHV = queryHV + productOne

  trainQueryHV[i] = queryHV


testQueryHV = np.zeros([testArrNumRows, 10000])
# encodes each test HV
for i in range(0, testArrNumRows):
  queryHV = np.zeros([10000])
  for j in range(0, testArrNumCols - 1):
    levelOne = levelHV[j, testOverallArr[i, j]]
    productOne = levelOne * featureHV[j]
    queryHV = queryHV + productOne
    
  testQueryHV[i] = queryHV

print("Done creating encoded train/test HV")


#finish creating trainQueryHV
a = overallArr[:, arrNumCols - 1].reshape(-1, 1)
print(a.shape)
trainQueryHV = np.c_[trainQueryHV, a]


#Y is an array holding all the accuracies of each version of the model after retraining
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

    print(k)


print(np.sum(accuracy), len(accuracy))
print(np.sum(accuracy)/len(testOverallArr))
print(len(testOverallArr))
print(Y)



