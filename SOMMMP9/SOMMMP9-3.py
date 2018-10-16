#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 12:44:33 2018
@author: mpcr
"""

##self-organizing map##
#DESeq2 data for MMP9 t-cell inhibititor conditions#


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import cv2
from scipy.misc import bytescale
#get the data without the first column as integers
#filename = 'DE0.csv'
filename='/home/mpcr/Desktop/Rachel/SparseCoding/DE0.csv'
data = np.genfromtxt(filename, delimiter=',', missing_values='NA', filling_values=1, usecols=range(1,7))
removeNA = data[:, -1] != 1
data = data[removeNA, :]
print(data.shape)
#get the data with the gene names in first column as a string
namedata = np.genfromtxt(filename, delimiter=',', dtype=str, usecols=0)
namedata = namedata[removeNA]
namedata = namedata[:] 
#namedata.shape = [namedata, 1]
print(namedata.shape)
#normalize the data
data = np.array(data, dtype=np.float64)
data -= np.mean(data, 0)
data /= np.std(data, 0)
print(data.shape)
#specify columns to be seperate inputs
n_in = data.shape[1]
#make the weights
w = np.random.randn(3, n_in) * 0.1
#hyperparameters
lr = 0.025
n_iters = 500
#do the training and show the weights on the nodes with cv2
for i in range(n_iters):
    randsamples = np.random.randint(0, data.shape[0], 1)[0] 
    rand_in = data[randsamples, :] 
    difference = w - rand_in
    dist = np.sum(np.absolute(difference), 1)
    best = np.argmin(dist)
    w_eligible = w[best,:]
    w_eligible += (lr * (rand_in - w_eligible))
    w[best,:] = w_eligible
    #cv2.namedWindow('weights', cv2.WINDOW_NORMAL)
    cv2.imshow('weights', bytescale(w))
    cv2.waitKey(100)
###############################################################################

#validation query
node1 = w[0, :]
node2 = w[1, :]
node3 = w[2, :]
difference1 = node1 - data
dist1 = np.sum(np.abs(difference1), 1)
dist1array = np.asarray(dist1)
dist1array.shape = (7905, 1)
difference2 = node2 - data
dist2 = np.sum(np.abs(difference2), 1)
dist2array = np.asarray(dist2)
dist2array.shape = (7905, 1)
difference3 = node3 - data
dist3 = np.sum(np.abs(difference3), 1)
dist3array = np.asarray(dist3)
dist3array.shape = (7905, 1)
#find the index in dist array that's lowest difference from representation node
top1 = np.argmin(dist1)
top2 = np.argmin(dist2)
top3 = np.argmin(dist3)
print(dist1[top1])
#add gene names to dist
dataname = []
for name in namedata:
    input = name
    input = input.lower()
    numbername = []
    for character in input:
        number = ord(character) 
        numbername.append(number)
    dataname.append(numbername)
len(dataname)#shape grew by one????? (7905 to 7906)????????????
finalnames = []
for element in dataname:
    numbername = int("".join(map(str, element)))
    type(numbername)
    finalnames.append(numbername)
len(finalnames)   
namesarray = np.asarray(finalnames)
namesarray.shape = (7905, 1)
print(namesarray.shape)
d1d = []
d1d = np.concatenate((namesarray, dist1), axis=1)
print(d1d.shape)
print(d1d[top1])

##convert queried gene-number-name back into gene-letter-name
#lettername = []
#for finalnames in data[:, 0]:
#    letters = chr(finalnames)
#    lettername.append(letters)
#print(lettername.shape)
#print(lettername)


ans = []
#ans = np.concatenate((namedata, dist2), axis=1)
#ans = np.append(namedata, dist2, axis=1)
#print(ans.shape)





###plot the data in 3d topology graph
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#Axes3D.plot_surface(X, Y, Z)

###convert queried gene-number-name back into gene-letter-name
#data *= np.std(data, 0)
#data += np.mean(data, 0)
#lettername = []
#for finalnames in data[:, 0]:
#    letters = chr(finalnames)
#    lettername.append(letters)
#print(lettername.shape)
#print(lettername)

###############################################################################
#show the weights for the datafile
print (filename)
print (w)

  
































  
    