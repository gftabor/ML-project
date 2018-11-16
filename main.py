#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 14:08:12 2018

@author: gtabor
"""

import perceptron
import preProcessing
import pickle
import numpy as np
import sys
def readExamples(folder,files):
    examples = []
    for file in files:
        f = open(folder+ file)
        lines = f.readlines()
        for line in lines:
            features = line.split(' ')
            example = []
            example.append([int(features[0])])
            for feature in features[1:]:
                data = feature.split(':')
                data = [int(data[0]),float(data[1])]
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
rawFolder = 'raw-data/'
files = ['data.train']
test_files = ['data.test']
devel_files = ['data.eval.anon']
devel_id = 'data.eval.anon.id'
vocab = 'vocab'




currentFolder = folder + cvFolder

train = readExamples(currentFolder,files)
test = readExamples(currentFolder,test_files)
devel = readExamples(currentFolder,devel_files)


#call these to generate new remap
count = preProcessing.findWordCount([train,test,devel])
#raw = folder + rawFolder
#debug = preProcessing.findSynonms(raw,vocab,count)
remapFiles = ['remapping_Rcheck_2.data','remapping_Rcheck_3.data','remapping_Rcheck_4.data']
argument = sys.argv[1]
print(argument)
print(sys.argv)
remapFile = remapFiles[int(argument)]
print(remapFile)
with open(remapFile, 'rb') as filehandle:  
    # read the data as binary data stream
    remapList = pickle.load(filehandle)
    
remap_dictionary={x[0]:x[1] for i,x in enumerate(remapList)}

rates = np.array(range(10))/10.0
rates = np.linspace(1,5,6)
#rates = [1,0.1,0.01]

rateModifierFunction = perceptron.decreasingRate

normalBestWeights = perceptron.performFullQuestion(rates,train,test,rateModifierFunction,[0], False)

preProcessing.applyRemap(train,remap_dictionary)
preProcessing.applyRemap(test,remap_dictionary)
preProcessing.applyRemap(devel,remap_dictionary)

remapBestWeights = perceptron.performFullQuestion(rates,train,test,rateModifierFunction,[0], False)


perceptron_Labels = perceptron.predict_all_labels(remapBestWeights,devel)

writeAnswers(currentFolder,devel_id,perceptron_Labels,remapFile + 'Perceptron.csv')



