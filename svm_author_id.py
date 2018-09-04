#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
"""
    Create the optimized classifier (Highest accuracy) and fit it with our training sets
"""
from sklearn import svm
from sklearn.metrics import accuracy_score
#labels_train = labels_train[:len(labels_train)/100]
#features_train = features_train[:len(features_train)/100]
clf=svm.SVC(kernel='rbf', C=10000)
t0 = time()
clf.fit(features_train, labels_train)
print ("training time:", round(time()-t0, 3), "s")
pred = clf.predict(features_test)
print ("predicting time:", round(time()-t0, 3), "s")
accuracy = accuracy_score(pred,labels_test)
print ("the accuracy of this classifier is", accuracy)

"""
    Extract predicitions for element 10 of the test set, the 26th and the 50th
"""
pred1 = labels_test[10]
pred2 = labels_test[26]
pred3 = labels_test[50]
print("the prediction of elements 10, 26 and 50 of the test set are, respectively", pred1, pred2, pred3)

"""
There are over 1700 test events how many are predicted to be in the 'Chris' (1) class?
"""
chris_occ=0
for i in range(len(pred)):
    if pred[i]==1:
        chris_occ+=1

print ("There are {} predicted to be in the Chris class".format(chris_occ))
#########################################################
