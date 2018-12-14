#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 14:10:49 2018

@author: tabor
"""

import os
os.environ["TFHUB_CACHE_DIR"] = './module'
import time
import tensorflow as tf
import pickle

import tensorflow_hub as hub
import numpy as np
class embedStuff():
    def __init__(self):
        # embed 2 scales better

        embed2 = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2", trainable=False)
        # embed 3 is transformer model. scales more poorly
        embed3 = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3", trainable=False)
        self.sess = tf.InteractiveSession()
        
        self.input_ph = tf.placeholder(tf.string, shape=[None]) # you will feed a paragraph or list of paragraph into this when you sess.run

        self.message_enc2 = embed2(self.input_ph)
        self.message_enc3 = embed3(self.input_ph)
        
        self.feature_ph = tf.placeholder(tf.float32, shape=[None,512]) 
        self.label_ph = tf.placeholder(tf.float32, shape=[None,1])
        
        ## neural network
        first = tf.layers.dense(self.feature_ph, 128, activation=tf.nn.relu) 
        second = tf.layers.dense(first, 128, activation=tf.nn.relu)
        
        self.output = tf.layers.dense(second, 1, activation=tf.nn.sigmoid)

        #self.cost = tf.nn.sigmoid_cross_entropy_with_logits(labels=label_ph, logits=self.output)
        self.cost = tf.losses.mean_squared_error(self.label_ph,self.output)
        self.train_op = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(self.cost)
        
        self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

    def preProcess(self,text):
        #2 or 3
        #print(text)
        out = self.sess.run(self.message_enc3, {self.input_ph: text})
        return out
    def preProcessBatch(self,text,batch):
        batches = (len(text) / batch) + 1
        splits = np.array_split(np.array(text),batches)
        feature_set = None
        for split in splits:
            features = self.preProcess(split)
            if(feature_set is None):
                feature_set = features
            else:
                feature_set = np.concatenate((feature_set,features))
        return feature_set
    def bulkPreProcess(self,folder):
        files = os.listdir(folder)
        
        print(files)
        for file in files:
            if(file[0] == 'd'):
                print('saw data')
                continue
            (text,labels) = readRawAmazonFiles(folder,[file])
            features = self.preProcessBatch(text,1000)
            with open(folder + 'data/' +str(file) +'.data', 'wb') as filehandle:  
                # store the data as binary data stream
                pickle.dump((labels,features), filehandle)
    def bulkTrain(self,folder,test_features,test_labels):
        files = os.listdir(folder)
        accuracies = []
        for i in range(20):
            np.random.shuffle(files) #for good science
            start = time.clock()
            for file in files:

                with open(folder+file, 'rb') as filehandle:  
                    # read the data as binary data stream
                    (labels,features) = pickle.load(filehandle)

                self.trainNN(labels,features)


            test_ = np.round(self.evaluateNN(test_features)).reshape((-1))
            a = test_ - np.array(test_labels)
            score = np.sum([a==0]) / 12500.
            print(str(time.clock() - start) + ' ' + str(score))
            accuracies.append(score)
            
        return accuracies
    def fullyTrainNN(self,labels,text,test_features,test_labels):
        accuracies = []
        for i in range(100):
            self.trainNN(labels,text)
                
            test_ = np.round(self.evaluateNN(test_features)).reshape((-1))
            a = test_ - np.array(test_labels)
            score = np.sum([a==0]) / 12500.
            accuracies.append(score)
        return accuracies
    def trainNN(self,labels,features):
        labels = np.asarray(labels).astype('float32').reshape((-1,1))        
        for i in range(10):
           _, loss = self.sess.run([self.train_op, self.cost], {self.feature_ph: features,self.label_ph: labels})
    def evaluateNN(self,features):
        labels = self.sess.run(self.output,{self.feature_ph: features})
        return labels
        
        
def readRawFiles(folder,files,normalData,banNegativeLabels = False):
    lines = []
    for file in files:
        f = open(folder+ file)
        lines += f.readlines()
    labels = []
    for index in range(len(lines)):
        label = normalData[index][0][0]
        if(banNegativeLabels and label < 0):
            label = 0
        labels.append(label)
    return (lines,labels)
def readRawAmazonFiles(folder,files):
    text = []
    labels = []
    for file in files:
        f = open(folder + file)
        lines = f.readlines()
        for line in lines:
            splits = line.split(' ',1)
            text.append(splits[1])
            if(splits[0] == '__label__2'):
                label = 1
            if(splits[0] == '__label__1'):
                label = 0
            labels.append(label)
    return (text,labels)

                    
def embedRawFiles(folder,files,normalData,embedder):
    (lines,labels) = readRawFiles(folder,files,normalData)
    for index in range(len(lines)):
        line = lines[index]
        label = labels[index]
        example = []
        example.append([int(label)])
        features = embedder.preProcess([line])
        count = 1
        for feature in features[1:]:
            data = [int(count),float(feature)]
            count +=1
            example.append(data)
        if(example[0][0]== 0):
            example[0][0] = -1
        normalData[index] = example
