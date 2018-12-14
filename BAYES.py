#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 14:08:12 2018

@author: gtabor
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle

def findWordCount(exampleSets):
    count1 = np.zeros(80000)
    count2 = np.zeros(80000)
    for examples in exampleSets: #train test devel
        for example in examples: #train
            if(example[0][0] == 1):
                for item in example[1:]: #
                    feature = item[0]
                    value = item[1]
                    count1[feature] += value
                count1[0] += 1    
            if(example[0][0] == -1):
                for item in example[1:]: #
                    feature = item[0]
                    value = item[1]
                    count2[feature] += value
                count2[0] += 1  
    return (count1,count2)

def sign(value):
    if(value >= 0):
        return 1
    else:
        return -1
def compute_example(weights,example):
    if type(weights) == type([1,2,3]) or len(weights.shape) == 1:

        y = weights[0]
        for index in range(len(example)):
            feature = example[index][0]
            value = example[index][1]
            y += weights[feature] * value
        return y
    else:
        posProb = weights[0][0]
        negProb = weights[0][1]
        for index in range(len(example)):
            feature = example[index][0]
            value = example[index][1]
            posProb = posProb * weights[feature][0]
            negProb = negProb * weights[feature][1]
        if(posProb > negProb):
            return 1.0
        else:
            return -1.0

def get_score(weights,examples):
    TP = 0.0001
    FP = 0.0001
    FN = 0.0001
    labels = []
    for example in examples:
        label = sign(compute_example(weights,example[1:]))
        labels.append(label)
        correctLabel = example[0][0]
        if( label == 1 and correctLabel == -1):
            FP += 1
        if( label == -1 and correctLabel == 1):
            FN += 1
        if( label == 1 and correctLabel == 1):
            TP += 1
    p = TP/(TP + FP)
    r = TP/(TP + FN)
    print("precision " + str(p)) 
    print("recall " + str(r)) 

    return (2 * (p * r) / ( p + r),labels)

def NaiveBayes(train,test,learningRate):
    trainSet = train[:] #copy to avoid shuffling issues
    testSet = test[:]
    weights = np.zeros((80000,2))
    
    (countPos,countNeg) = findWordCount([train])
    
    total = countPos[0] + countNeg[0]
    pyPos = countPos[0] / float(total)
    pyNeg = countNeg[0] /float(total)
    
    for index in range(len(countPos)):
        if(index == 0 ):
            weights[index,0] = pyPos
            weights[index,1] = pyNeg
        else:
            weights[index,0]  = (countPos[index] + learningRate) / (2 * learningRate + countPos[0])
            weights[index,1] = (countNeg[index] + learningRate) / (2 * learningRate + countNeg[0])
        
    (score,labels) = get_score(weights,testSet)
    print(score)
    return labels


