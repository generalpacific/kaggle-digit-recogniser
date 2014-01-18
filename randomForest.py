	
import re	
import csv 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from datetime import datetime

def log(string):
	print str(datetime.now()) + " " + string

if __name__ == '__main__':	

	log("Reading Train Data")
	
	train = csv.reader(open('train.csv','rb'))
	header = train.next()
	
	######READING TRAIN DATA################	
	train_data=[]
	for row in train:
	        train_data.append(row)
	
	train_data = np.array(train_data)
	
	log("DONE Reading Train Data")
	
	features = train_data[0::,1::]
	result = train_data[0::,0]
	log("DONE Preprocessing Train Data")

	log("Fitting Train Data")
	adaBoost = AdaBoostClassifier(RandomForestClassifier(n_estimators = 1000),
                         algorithm="SAMME",
                         n_estimators=200)
	
	adaBoost = adaBoost.fit(features,result)
	log("DONE Fitting Train Data")

	######READING TEST DATA################	
	log("Reading Test Data")
	test = csv.reader(open('test.csv','rb'))
	header = test.next()
	
	test_data=[]
	for row in test:
	        test_data.append(row)
	test_data = np.array(test_data)
	log("DONE Reading Test Data")
	
	log("DONE Preprocessing Test Data")
	
	log("Predicting Test Data")
	Output = adaBoost.predict(test_data)
	
	np.savetxt("adaBoostRandomForest.csv",Output,delimiter=",",fmt="%s")	
	log("DONE Predicting Test Data")
