#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 14:08:12 2018

@author: gtabor
"""

import perceptron

import rawData
import pickle
import numpy as np
import sys
import time
import matplotlib.pyplot as plt

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
def synonym(train,test,devel,raw):
    import preProcessing
    #call these to generate new remap
    vocab = 'vocab'

    #count = preProcessing.findWordCount([train,test,devel])
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
    preProcessing.applyRemap(train,remap_dictionary)
    preProcessing.applyRemap(test,remap_dictionary)
    preProcessing.applyRemap(devel,remap_dictionary)
def DNN(train,test,devel,raw):

    embed = rawData.embedStuff()
    start = time.clock()
    train = rawData.embedRawFiles(raw,trainRaw,train,embed)
    print('took ' + str(time.clock()- start) )
    test = rawData.embedRawFiles(raw,testRaw,test,embed)
    print('took ' + str(time.clock()- start))
    devel = rawData.embedRawFiles(raw,evalRaw,devel,embed)
    print('took ' + str(time.clock()- start))
    
#declare variables
folder = 'movie-ratings/'
cvFolder = 'data-splits/'
rawFolder = 'raw-data/'
files = ['data.train']
test_files = ['data.test']
devel_files = ['data.eval.anon']
testRaw = ['test.rawtext']
trainRaw = ['train.rawtext']
evalRaw = ['eval.rawtext']

devel_id = 'data.eval.anon.id'

mainFolder = folder + cvFolder
raw = folder + rawFolder


train = readExamples(mainFolder,files)
test = readExamples(mainFolder,test_files)
devel = readExamples(mainFolder,devel_files)

#How to preprocess
#synnonym(train,test,devel,raw)
#DNN(train,test,devel,raw)




#perceptron run
#rates = [1,0.1,0.01]
#rateModifierFunction = perceptron.sameRate
#normalBestWeights = perceptron.performFullQuestion(rates,train,test,rateModifierFunction,[0], False)
#labels = perceptron.predict_all_labels(normalBestWeights,devel)

embed = rawData.embedStuff()
(train_lines,train_labels) = rawData.readRawFiles(raw,trainRaw,train,True)
(test_lines,test_labels) = rawData.readRawFiles(raw,testRaw,test,True)
(eval_lines,eval_labels) = rawData.readRawFiles(raw,evalRaw,devel,True)
train_features = embed.preProcessBatch(train_lines,1000)
test_features = embed.preProcessBatch(test_lines,1000)
eval_features = embed.preProcessBatch(eval_lines,1000)

losses = embed.trainNN(train_labels,train_features)
plt.plot(losses)
test_ = embed.evaluateNN(test_features)
labels = embed.evaluateNN(eval_features)



writeAnswers(mainFolder,devel_id,labels,'DNN' + '.csv')



