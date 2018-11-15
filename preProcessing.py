#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 13:17:32 2018

@author: tabor
"""

from thesaurus import Word
import numpy as np
def findWordCount(exampleSets):
    count = np.zeros(75000)
    for examples in exampleSets: #train test devel
        for example in examples: #train
            for item in example[1:]: #
                feature = item[0]
                value = item[1]
                count[feature] += value
    return count

def findSynonms(folder,fileName):
    f = open(folder+ fileName)
    lines = f.readlines()

    for i in range(len(lines)):
        word = lines[i][:-1]
        new_instance = Word(word)
        a = new_instance.synonyms()
        print(i)
#        print(word)
#        print(new_instance.synonyms())

    