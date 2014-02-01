
# This code will compare different algorithms for the dataset
# To add a new algorithm use the preprocessed data and write a new method and call the method in the end.

import re	
import csv 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from datetime import datetime
from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

# Add time along with the log
def log(logname,string):
        print str(datetime.now()) + "\t"  + logname + "\t" + string

##################################################
# METHODS FOR DIFFERENT ALGORITHMS
#
# Tips to add new algorithm:
#	1. Copy the following random forest code. 
#	2. Change the place holders accordingly
##################################################

def randomforest(trainfeatures,trainlabels,testfeatures):
	RandomForest = RandomForestClassifier(n_estimators = 1000)
	return runalgorithm(RandomForest,trainfeatures,trainlabels,testfeatures)

def decisiontree(trainfeatures,trainlabels,testfeatures):
	tree = DecisionTreeClassifier(random_state = 1000)
	return runalgorithm(tree,trainfeatures,trainlabels,testfeatures)

def adaboost(trainfeatures,trainlabels,testfeatures):
	adaBoost = AdaBoostClassifier(RandomForestClassifier(n_estimators = 1000),
                         algorithm="SAMME",
                         n_estimators=200)
	return runalgorithm(adaBoost,trainfeatures,trainlabels,testfeatures)

# Generic code for running any algorithm called from above algorithms
def runalgorithm(algorithm,trainfeatures,trainlabels,testfeatures):
	logname = runalgorithm.__name__
	algorithmName = algorithm.__class__.__name__
	
	log(logname,algorithmName + " Fitting train data")
        algorithm = algorithm.fit(trainfeatures,trainlabels)
	log(logname,algorithmName + " DONE Fitting train data")
	
	log(logname,algorithmName + " Scoring train data")
	scores = cross_val_score(algorithm, trainfeatures, trainlabels)
	score = scores.mean()
	score = str(score)
	log(logname,algorithmName + " Score : " + score)
	log(logname,algorithmName + " DONE Scoring train data")
	
	log(logname,algorithmName + " Predicting test data")
	Output = algorithm.predict(testfeatures)	
	log(logname,algorithmName + " DONE Predicting test data")
	writeFile = algorithmName + ".csv"
	log(logname,algorithmName + " Writing results to " + writeFile)
	np.savetxt(writeFile,Output,delimiter=",algorithmName + " ,fmt="%s")
	log(logname,algorithmName + " DONE Writing results to " + writeFile)
	return score

##################################################
# MAIN METHOD
##################################################
if __name__ == '__main__':	
	logname = "__main__"
	
	log(logname,"Reading Train Data")
	
	train = csv.reader(open('train.csv','rb'))
	header = train.next()
	
	######READING TRAIN DATA################	
	train_data=[]
	for row in train:
	        train_data.append(row)
	
	train_data = np.array(train_data)
	
	log(logname,"DONE Reading Train Data")
	
	trainfeatures = train_data[0::,1::]
	trainlabels = train_data[0::,0]

	log(logname,"length of train:"+str(len(trainfeatures[0])))	
	log(logname,"applying pca");
	pca = PCA(n_components=33)

	trainfeatures = pca.fit_transform(trainfeatures)
	log(logname,"DONE applying pca");
	log(logname,"length of train:"+str(len(trainfeatures[0])))	
	log(logname,"DONE Preprocessing Train Data")

	######READING TEST DATA################	
	log(logname,"Reading Test Data")
	test = csv.reader(open('test.csv','rb'))
	header = test.next()
	
	test_data=[]
	for row in test:
	        test_data.append(row)
	testfeatures = np.array(test_data)
	log(logname,"DONE Reading Test Data")
	
	log(logname,"length of test:"+str(len(testfeatures[0])))	
	log(logname,"applying pca");

	testfeatures = pca.fit_transform(testfeatures)
	log(logname,"DONE applying pca");
	log(logname,"length of test:"+str(len(testfeatures[0])))	
	log(logname,"DONE Preprocessing Test Data")
	
	####################### TRAIN AND TEST ##########################

	scores = {}

	log(logname,"Calling Random Forest")
	score = randomforest(trainfeatures,trainlabels,testfeatures)
	scores['Random Forest'] = score
	log(logname,"DONE WITH Random Forest")
#
#	log(logname,"Calling AdaBoost")
#	score = adaboost(trainfeatures,trainlabels,testfeatures)
#	scores['AdaBoost'] = score
#	log(logname,"DONE WITH AdaBoost")
	
	log(logname,"Calling Decision Tree")
	score = decisiontree(trainfeatures,trainlabels,testfeatures)
	scores['Decision Tree'] = score
	log(logname,"DONE WITH Decision Tree")

	print "\nSCORES\n"
	for k, v in scores.iteritems():
		print k + "\t" + v
		
