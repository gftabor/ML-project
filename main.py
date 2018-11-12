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
            examples.append(example)
    return examples

#declare variables
folder = 'movie-ratings/'
cvFolder = 'data-splits/'
files = ['data.train']
test_files = ['data.test']
devel_files = ['diabetes.dev']

currentFolder = folder + cvFolder

train = readExamples(currentFolder,files)
test = readExamples(currentFolder,test_files)


rates = [1,0.1,0.01]


output = perceptron.trainAndEvaluate(train,test,rates[0],perceptron.sameRate,10,0,average = False)




#performFullQuestion(rates,sameRate)
