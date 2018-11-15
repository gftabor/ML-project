#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 13:17:32 2018

@author: tabor
"""

from thesaurus import Word
import numpy as np
from multiprocessing import Pool
import pickle


part = ['adj','adv','contraction','conj','determiner',
        'interj','noun','prefix','prep','pron','verb',
        'abb']
def findWordCount(exampleSets):
    count = np.zeros(75000)
    for examples in exampleSets: #train test devel
        for example in examples: #train
            for item in example[1:]: #
                feature = item[0]
                value = item[1]
                count[feature] += value
    last_real = np.argmin(count[1:])
    real_count = count[:last_real+1]
    return count
def applyRemap(dataset,remap):
    for example in dataset:
        for featureIndex in range(len(example[1:])):
            feature = example[1:][featureIndex][0]
            value = example[1:][featureIndex][1]
            if(feature in remap):#worth remapping
                newFeature = remap[feature]
                for otherFeatureIndex in range(len(example[1:])):
                    if(otherFeatureIndex == featureIndex): #if looking at same point twice, shouldnt happne was dumb
                        continue
                    otherFeature = example[1:][otherFeatureIndex][0]
                    otherValue = example[1:][otherFeatureIndex][1]
                    if(newFeature == otherFeature): #if found 
                        example[1:][otherFeatureIndex][1] += value
                        example[1:][featureIndex][1] = 0
                        break
                example[1:][featureIndex][0] = newFeature #if new feature isnt in example change featureID
                        
            else:
                continue
def findWordInDataset(word_Dictionary,synonymSets,word,count):
    total = 0
    for synonymSet in synonymSets:
        total += len(synonymSet)
    if(total == 0):
        return False
    for synonymSet in  synonymSets:
        for synonym in  synonymSet:
            if(synonym == word):
                continue
            if(synonym in word_Dictionary):
                occurances = count[word_Dictionary[synonym]]
                if(occurances > 5):
                    print(word + ' ' + synonym + ' ' + str(word_Dictionary[synonym]))
                    #index of synonym
                    return word_Dictionary[synonym]
                else:
                    #print('too small ' + synonym + ' ' + str(occurances))
                    a =5
    return False
def parallel(inputs):
    (word, word_Dictionary,count,chosen_index) = inputs
    new_instance = Word(word)
    synonyms = new_instance.synonyms('all',relevance = [3],partOfSpeech=part)
    response = findWordInDataset(word_Dictionary,synonyms,word,count)
    if(response == False):
        synonyms = new_instance.synonyms('all',relevance = [2],partOfSpeech=part)
        response = findWordInDataset(word_Dictionary,synonyms,word,count)
        if(response == False):
            synonyms = new_instance.synonyms('all',relevance = [1],partOfSpeech=part)
            response = findWordInDataset(word_Dictionary,synonyms,word,count)
    return(chosen_index,response)

def findSynonms(folder,fileName,count):
    f = open(folder+ fileName)
    #dumb so that counts line up with correct word
    lines = f.readlines()
    lines.insert(0,'___asdfasdf')
    lines = [s.strip('\n') for s in lines]
    word_Dictionary={x:i for i,x in enumerate(lines)}
    indices = np.argsort(count)
    index = 1 #skip first its 0
    inputSet = []
    cuttoff = 2
    while(count[indices[index]]<cuttoff):
        chosen_index = indices[index]
        word = lines[chosen_index]
        inputSet.append((word, word_Dictionary,count,chosen_index))
        index +=1
            
    p = Pool(20)

    Remapping = p.map(parallel,inputSet)
    remap = [s for s in Remapping if s[1] != False]
    with open('remapping'+str(cuttoff)+'.data', 'wb') as filehandle:  
    # store the data as binary data stream
        pickle.dump(remap, filehandle)
    return remap
    
    
    
        

    