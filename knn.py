# Example of kNN implemented from Scratch in Python
 
import csv
import random
import math
import operator
import numpy as np
 
def loadDataset(filename, split, trainingSet=[] , testSet=[]):
	with open(filename, 'rb') as csvfile:
	    lines = csv.reader(csvfile)
	    dataset = list(lines)
	    dataset=normalize(dataset)
	    for x in range(len(dataset)-1):
	        for y in range(4):
	            dataset[x][y] = float(dataset[x][y])
	        if random.random() < split:
	            trainingSet.append(dataset[x])
	        else:
	            testSet.append(dataset[x])
 
def normalize(data):
    data =  np.array(data).astype(np.float)
    maks = np.amax(data,axis=0)
    mini = np.amin(data,axis=0)
    for x in range(len(data)-1):
        for y in range(8):
            data[x][y]=(data[x][y]-mini[y])/(maks[y]-mini[y])
    return data
 
def euclideanDistance(instance1, instance2, length):
	distance = 0     
	for x in range(length):
		v1=float(instance1[x]); v2=float(instance2[x]);
		distance += pow((v1 - v2), 2)
	return math.sqrt(distance)
 
def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors
 
def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]
 
def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def knn(trainingSet, testSet, k, predictions):
	for x in range(len(testSet)):
		neighbors = getNeighbors(trainingSet, testSet[x], k)
		result = getResponse(neighbors)
		predictions.append(result)
	accuracy = getAccuracy(testSet, predictions)
	return accuracy

def pso(P,n,trainingSet,testSet,predictions):
	P=[]
	p = 1
	for x in range (n):                    
		P.append(p)
		p+=2  
	Pbest=P
	gb = 0
	c1 = c2 = 1 
	V = []
	for x in range(n):
		V.append(0)
	while (1):
		for p in range(len(P)):
			fp=knn(trainingSet, testSet, P[p], predictions)
			if fp > knn(trainingSet, testSet, Pbest[p], predictions):
				Pbest[p] = P[p]
			if fp > gb:
				gb = fp
				Gbest = P[p]               
		for p in range(len(P)):
			V[p]=V[p]+c1*random.random()*(Pbest[p]-P[p])+c2*random.random()*(Gbest-P[p])
			P[p]=P[p]+V[p]                        
       

def main():
	# prepare data
	trainingSet=[]
	testSet=[]
	P=[]
	split = 0.67
	n = int(input('enter number of Particle: '))
	loadDataset('pima-indians-diabetes.data', split, trainingSet, testSet)
	print( 'Train set: ' + repr(len(trainingSet)))
	print( 'Test set: ' + repr(len(testSet)))
	# generate predictions
     
	predictions=[]
	pso(n,trainingSet, testSet, predictions)
     
	
main()