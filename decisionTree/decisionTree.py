#!/usr/bin/python

'''
Decision tree -- lazy just use sklearn
Use UCI iris database
'''

from sklearn import tree
import graphviz
import random

irisDB = open('./databases/iris/iris.data')
lines = irisDB.readlines()
x = []; y = [0]; lastTag = ''
tagList = []
for line in lines :
    splitLine = line.split(',')
    dat = []
    for i in range(4) :
        dat.append(float(splitLine[i]))
    x.append(dat)
    if lastTag == '' :
        lastTag = splitLine[4]
        tagList.append(lastTag)
    elif splitLine[4] != lastTag :
        y.append(y[len(y)-1] + 1)
        lastTag = splitLine[4]
        tagList.append(lastTag)
    else :
        y.append(y[len(y)-1])
trainX = []; trainY = []; testX = []; testY = []
for i in range(len(x)) :
    if random.random() > 0.3 :
        trainX.append(x[i])
        trainY.append(y[i])
    else :
        testX.append(x[i])
        testY.append(y[i])
print('Training set:', len(trainX), 'Test set:', len(testX))

# Train the tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(trainX, trainY)
pred = clf.predict(testX)
correct = sum([1 for i in range(len(pred)) if pred[i] == testY[i]])
print('Correct:', correct / len(pred))

# Draw the tree
dot_data = tree.export_graphviz(clf, out_file=None,
                      #feature_names=iris.feature_names,
                      class_names=tagList,
                      filled=True, rounded=True,
                      special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("iris")
