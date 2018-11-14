#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 14:08:12 2018

@author: gtabor
"""

import perceptron
def readExamples(folder,files):
    examples = []
    for file in files:
        f = open(folder+ file)
        lines = f.readlines()
        for line in lines:
            features = line.split(sep=' ')
            example = []
            example.append([int(features[0])])
            for feature in features[1:]:
                data = feature.split(':')
                data = (int(data[0]),float(data[1]))
                example.append(data)
            if(example[0][0]== 0):
                example[0][0] = -1
            examples.append(example)
    return examples
def writeAnswers(folder,file,labels,name):
    newFile = open(name,"w")
    f = open(folder + file)
    lines = f.readlines()
    newFile.write('example_id,label\n')
    for index in range(len(lines)):
        label = labels[index]
        if(label < 0):
            label = 0
        newString = lines[index][:-1] +',' + str(label) + lines[index][-1]
        newFile.write(newString)
    newFile.close
#declare variables
folder = 'movie-ratings/'
cvFolder = 'data-splits/'
files = ['data.train']
test_files = ['data.test']
devel_files = ['data.eval.anon']
devel_id = 'data.eval.anon.id'


currentFolder = folder + cvFolder

train = readExamples(currentFolder,files)
test = readExamples(currentFolder,test_files)
devel = readExamples(currentFolder,devel_files)


rates = [1,0.1,0.01,0.001]


#output = perceptron.trainAndEvaluate(train,test,rates[0],perceptron.sameRate,10,0,average = False)

bestWeights = perceptron.performFullQuestion(rates,train,test,perceptron.sameRate,[0], False)
perceptron_Labels = perceptron.predict_all_labels(bestWeights,devel)
writeAnswers(currentFolder,devel_id,perceptron_Labels,'perceptron.csv')

#performFullQuestion(rates,sameRate)
