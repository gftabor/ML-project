#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 01:11:14 2018

@author: gtabor
"""

import BAYES as main
import numpy as np
import math

def run(trainSet,testSet,depth):
    root = Node(range(80000),0)
    train(root,trainSet,depth)
        

    labels = []
    for index in range(len(testSet)):
        labels.append(predict(root,testSet[index]))
    return labels
class Node:
    def __init__(self,remainingAttributes,depth):
        self.remaining = remainingAttributes
        self.depth = depth
        self.label = None
    def setAtt(self,att):
        self.attribute = att
    def setLabel(self,label):
        self.label = label
    def addChildren(self,options):
        self.children = options
    def __str__(self):
        if self.label is None:
            return str(self.attribute)
        else:
            return 'label ' + str(self.label)
def entropyMath(set,sum):
    if(set < 1.0):
        return 0
    sum = abs(sum)
    percent = set/sum
    value =  -1* percent * math.log(percent,2)
    return value
def weighted_Entropy(subset,featureIndex):
    (count1,count2) = main.findWordCount([subset])
    total = count1[0] + count2[0]
    weighted = 0
    
    #feature = 1
    posLabels = count1[featureIndex]
    negLabels = count2[featureIndex]
    subtotal = posLabels + negLabels
    entropy = 0    
    entropy += entropyMath(posLabels,subtotal)
    entropy += entropyMath(negLabels,subtotal)
    if(subtotal > 0.5):
        weighted += (subtotal / total) * entropy
    
    #feature = 0
    posLabels = count1[0] - count1[featureIndex]
    negLabels = count2[0] - count2[featureIndex]
    subtotal = posLabels + negLabels
    entropy = 0    
    entropy += entropyMath(posLabels,subtotal)
    entropy += entropyMath(negLabels,subtotal)
    if(subtotal > 0.5):
        weighted += (subtotal / total) * entropy
    
    return entropy 
def Entropy(subset):
    #(count1,count2)
    counts = main.findWordCount([subset])
    total = counts[0][0] + counts[1][0]
    entropy = 0
    for i in range(len(counts)):
        percent = counts[i][0]/total
        entropy += -1* percent * math.log(percent,2)
    return entropy 
def getSubsets(examples,featureIndex):
    pos = []
    neg = []
    for example in examples:
        hasIt = hasIndex(example,featureIndex)
        if(hasIt):
            pos.append(example)
        else:
            neg.append(example)
    return (pos,neg)
def predict(node,example):
    if(node.label == None):
        if(hasIndex(example,node.attribute)):
            return predict(node.children[1],example)
        else:
            return predict(node.children[0],example)
    else:
        return node.label
def train(node,subset,maxDepth):
    keys = []
    gains = []
    removeList = []
    for key in node.remaining:
        inf_gain = weighted_Entropy(subset,key)
        if(inf_gain == -1):
            removeList.append(key)
        else:
            keys.append(key)
            gains.append(inf_gain)
    (count1,count2) = main.findWordCount([subset])

    if(node.depth >= maxDepth or count1[0] + count2[0] < 2):
        if(count1[0] > count2[0]):
            node.setLabel(1)
            return
        else:
            node.setLabel(-1)
            return    
    index = np.argmin(gains)
    #new node decides on that feature
    att = keys[index]
    node.setAtt(att)
    children = {}
    (posSubset,negSubset) = getSubsets(subset,att)
    #just to fit with 0 1 indexing
    subsets = [negSubset,posSubset]
    children = {}
    node.addChildren(children)

    for option in [0,1]:
        newAttributeList = []

        #tell node beneath not to include you in its list
        newAttributeList.extend(node.remaining)
        newAttributeList.remove(att)
        child = Node(newAttributeList,node.depth + 1)
        children[option] = child
        node.addChildren(children)

        train(child,subsets[option],maxDepth)

def hasIndex(example,featureIndex):
    for index in range(len(example)):
        feature = example[index][0]
        if(feature == featureIndex):
            return True
    return False


        
