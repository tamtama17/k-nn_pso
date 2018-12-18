import csv
import math
import operator
import numpy as np
import random

def loadDataset(filename, idx, trainingSet=[] , testSet=[]):
    dataset = list()
    with open(filename, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        for i in lines:
            dataset.append(i[1:])
        random.shuffle(dataset)
        dataset = normalisasi(dataset)
        fold = math.ceil(len(dataset)/10)+1
        for x in range(len(dataset)):
            if x >= fold*idx and x < (fold*idx)+fold:
                testSet.append(dataset[x])
            else:
                trainingSet.append(dataset[x])

def normalisasi(dataset):
    dataset = np.array(dataset).astype(float)
    maxim = np.amax(dataset, axis=0)
    minim = np.amin(dataset, axis=0)
    
    for i in range(len(dataset)):
        for j in range(len(dataset[i])-1):
            dataset[i][j] = (dataset[i][j] - minim[j]) / (maxim[j] - minim[j])
    
    return dataset

def cosineSimilarity(instance1, instance2, length):
    a = []
    b = []
    for x in range(length):
        a.append(instance1[x])
        b.append(instance2[x])
    x = np.array(a, dtype=float)
    y = np.array(b, dtype=float)
    dot_product = np.dot(x, y)
    norm_a = np.linalg.norm(x)
    norm_b = np.linalg.norm(y)
    return 1 - (dot_product/(norm_a*norm_b))

def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((float(instance1[x]) - float(instance2[x])), 2)
	return math.sqrt(distance)

def manhattanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += abs(float(instance1[x]) - float(instance2[x]))
    return distance

def getNeighbors(trainingSet, testInstance, k, method):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        if method == 1:
            dist = cosineSimilarity(testInstance, trainingSet[x], length)
        elif method == 2:
            dist = euclideanDistance(testInstance, trainingSet[x], length)
        elif method == 3:
            dist = manhattanDistance(testInstance, trainingSet[x], length)
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

def getResult(neighbors):
    total = 0
    for x in range(len(neighbors)):
        total += neighbors[x][-1]
    return float(total/len(neighbors))

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

def getRegAccuracy(testSet, predictions):
    accuracy = 0
    for x in range(len(testSet)):
        accuracy += abs(float(testSet[x][-1])-float(predictions[x]))/float(testSet[x][-1])
    return float(1-(accuracy/len(testSet))) * 100.0

def ambil_kunci(kamus,cari):
    for key in kamus:
        if kamus[key]==cari:
            return key
        
def k_nn(iter_k,method):
    accuracy = 0
    for x in range(10):
        trainingSet=[]
        testSet=[]
        loadDataset('zoo.data', x, trainingSet, testSet)
        
        #print 'Train set: ' + repr(len(trainingSet))
        #print 'Test set: ' + repr(len(testSet))
        predictions=[]
        for y in range(len(testSet)):
            neighbors = getNeighbors(trainingSet, testSet[y], iter_k, method)
            result = getResponse(neighbors)
            predictions.append(result)
        
        temp = getAccuracy(testSet, predictions)
        
        #print (repr(temp) + '%')
        accuracy = float(accuracy + temp)
        #print iter_k
    accuracy = accuracy/10
    return float(accuracy)

def pso(n, method):
	P=[]
	p = 1
	konv = 0
	rand = float(random.randint(1,10))/10
	memo=[None for v in range(999)]
	for x in range (n):
		P.append(p)
		p+=2

	Pbest=P
	gb = 0
	Gbest = 0
	c1 = c2 = 1
	V = []
	for x in range(n):
		V.append(0)
	iters = 0
	while (konv==0):
		print('Iter-'+ repr(iters))
		print('Gbest = '+repr(Gbest))
		iters+=1
		p=0
		for p in range(len(P)):
			if(memo[int(P[p])] is not None):
				fp = memo[int(P[p])]
			else:
				fp=k_nn(int(P[p]),method)
				memo[int(P[p])] = fp       
			Pbest[p] = P[p]
			print(repr(P[p])+' - '+repr(fp))
			if fp > gb:
				gb = fp
				Gbest = P[p]
		p=0
		for p in range(len(P)):
			V[p]=V[p]+c1*rand*(Pbest[p]-P[p])+c2*rand*(Gbest-P[p])
			print V
			if V[p]-math.floor(V[p])!=0:   #jika velocity bukan bilangan bulat
				if V[p]>0:                  # jika velocity positive
					V[p]=math.ceil(V[p])
				else:                       # jika velocity negative
					V[p]=math.floor(V[p])
			if V[p]%2!=0:
				if V[p]>0:                  # jika velocity positive
					V[p]=V[p]+1
				else:                       # jika velocity negative
					V[p]=V[p]-1

			P[p]=P[p]+int(V[p])
			if P[p]<1:
				P[p]=1

		p=0
		for p in range(len(P)-1):
			if P[p] != P[p+1]:
				konv=0
				break
			else:
				konv=1
		print('\n')

	return Gbest

def main():
    # prepare data
    
    print 'Implementasi K-NN dengan optimasi PSO dan dataset zoo.data'
    #k = 3
    j_partikel = input('Jumlah partikel = ')
    
    print '\nMetode yang ingin digunakan?'
    print '1. Cosine Similarity'
    print '2. Euclidean Distance'
    print '3. Manhattan Distance'
    method = input('Pilih metode = ')
    if (method != 1) and (method != 2) and (method != 3):
        print 'Input error'
        return 0
    print pso(j_partikel, method)
    print '\n'
main()