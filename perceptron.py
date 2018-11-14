#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 23:02:04 2018

@author: gtabor
"""
import numpy as np
import matplotlib.pyplot as plt
def sign(value):
    if(value >= 0):
        return 1
    else:
        return -1
def compute_example(weights,example):
    y = weights[0]
    for index in range(len(example)):
        feature = example[index][0]
        value = example[index][1]
        y += weights[feature] * value
    return y
def score_example(weights,example):
    val = compute_example(weights,example[1:])
    labelGuess = sign(val)
    if(labelGuess == example[0][0]):
        return 1
    else:
       # print(example[0][0])
        return 0
def get_score(weights,examples):
    score = 0
    for example in examples:
        score += score_example(weights,example)
    percentage = score/float(len(examples))
    return percentage
def learn_example(rate,weights,example,epsilon):
    val = compute_example(weights,example[1:])
    correctLabel = example[0][0]
   # print(str(labelGuess) + ' ' + str(correctLabel))
    if(correctLabel * val >= epsilon):
        return 0
    else:
        for feature in example:
            learn = rate  * correctLabel * feature[-1]
          #  print(learn)
            weights[feature[0]]  += learn
        weights[0] += rate*correctLabel
        return 1
def sameRate(count,epsilon,weights,example):
    return 1
def decreasingRate(count,epsilon,weights,example):
    return 1/(1.0 + count)
def aggressiveRate(count,epsilon,weights,example):
    y = score_example(weights,example)
    dot = 0
    for feature in example:
        dot += feature[-1]**2
    rate = (epsilon - y)/(dot + 1)
    return rate

def trainAndEvaluate(train,test,learningRate,rateModifierFunction,epoch,epsilon,average = False):
    np.random.seed(473)
    trainSet = train[:] #copy to avoid shuffling issues
    testSet = test[:]
    weights = np.random.rand(80000) * 0.2 -0.1
    weightSet = []
    count = 0
    updateCount = 0
    updateCountSet = []
    averageWeights = np.random.rand(80000) * 0
    for epoch in range(epoch):
        np.random.shuffle(trainSet)
        for example in trainSet:
            count += 1
            currentRate = learningRate * rateModifierFunction(count,epsilon,weights,example)
            updateCount += learn_example(currentRate,weights,example,epsilon)
            averageWeights = np.add(weights,averageWeights)
        if(average):
            saveWeights = list(averageWeights)
        else:
            saveWeights = list(weights)
        weightSet.append(saveWeights)
        updateCountSet.append(updateCount)
    score = get_score(saveWeights,testSet)
    return (score,weightSet,updateCountSet)

def performFullQuestion(rates,train,test,rateModfier,epsilons = [0],average = False):
    perEpsilonAccuracy = []
    bestRates = []
    for epsilon in epsilons:
        crossAccuracy = []
        for rate in rates:
            (score, weightSet,updateCountSet) = trainAndEvaluate(train,test,rate,rateModfier,10,epsilon,average)
            crossAccuracy.append(score)
        index = np.argmax(crossAccuracy)
        bestRate = rates[index]
        bestRates.append(bestRate)
        perEpsilonAccuracy.append(crossAccuracy[index])
        print(crossAccuracy)
    index = np.argmax(perEpsilonAccuracy)
    bestEpsilon = epsilons[index]
    bestRate = bestRates[index]
    bestAccuracy = perEpsilonAccuracy[index]
    print('cross accuracy ' + str(bestAccuracy))
    print('best rate ' + str(bestRate))
    print('best epsilon ' + str(bestEpsilon))
    
    
    
    (score, weightSet,updateCountSet) = trainAndEvaluate(train,test,bestRate,rateModfier,20,bestEpsilon,average)
    epochScores = []
    for weights in weightSet:
        score = get_score(weights,test)
        epochScores.append(score)
    index = np.argmax(epochScores)
    print('updates on learning algorithm ' +str(updateCountSet[index]))
    print('devel accuracy ' + str(epochScores[index]))

    bestWeights = weightSet[index]
    
    bestPerceptron = get_score(bestWeights,test)
    shitWeights = np.random.rand(80000) *0
    shitWeights[0] = -10
    negativeOnly = get_score(shitWeights,test)
    print('negativeOnly accuracy ' + str(negativeOnly))

    print('test accuracy ' + str(bestPerceptron))
    plt.plot(epochScores)
    plt.show()