#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 09:39:15 2018

@author: cats
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
filename='desktop/DE0.csv'
data = np.genfromtxt(filename, delimiter=',', missing_values='NA', skip_header=2, filling_values=1, usecols=range(1,7))
removeNA = data[:, -1] != 1
data = data[removeNA, :]
print(data.shape)
#get the data with the gene names in first column as a string
namedata = np.genfromtxt(filename, delimiter=',', skip_header=2,  dtype=str, usecols=0)
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
number_nodes = 3
w = np.random.randn(number_nodes, n_in) * 0.1
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
    cv2.namedWindow('weights', cv2.WINDOW_NORMAL)
    cv2.imshow('weights', bytescale(w))
    cv2.waitKey(100)  
 #wuery validation   
node1 = w[0, :]
node2 = w[1, :]
node3 = w[2, :]
difference1 = node1 - data
dist1 = np.sum(np.abs(difference1), 1)
difference2 = node2 - data
dist2 = np.sum(np.abs(difference2), 1)
difference3 = node3 - data
dist3 = np.sum(np.abs(difference3), 1)
#find the index in dist array that's lowest difference from representation node
top1 = np.argmin(dist1)
top2 = np.argmin(dist2)
top3 = np.argmin(dist3)
ans1 = namedata[top1]
ans2 = namedata[top2]
ans3 = namedata[top3]
ans = []
ans.append(ans1)
ans.append(ans2)
ans.append(ans3)

###plot the data in 3d topology graph
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#Axes3D.plot_surface(X, Y, Z)

#show the weights for the datafile
print (filename)
print (w)
print(ans)
###############################################################################

#get the data without the first column as integers
#filename = 'DE0.csv'
filename2='Desktop/DE2.csv'
data2 = np.genfromtxt(filename2, delimiter=',', missing_values='NA', skip_header=2, filling_values=1, usecols=range(1,7))
removeNA = data2[:, -1] != 1
data2 = data2[removeNA, :]
print(data2.shape)
#get the data with the gene names in first column as a string
namedata = np.genfromtxt(filename2, delimiter=',', skip_header=2, dtype=str, usecols=0)
namedata = namedata[removeNA]
namedata = namedata[:] 
#namedata.shape = [namedata, 1]
print(namedata.shape)
#normalize the data
data2 = np.array(data2, dtype=np.float64)
data2 -= np.mean(data2, 0)
data2 /= np.std(data2, 0)
print(data.shape)
#specify columns to be seperate inputs
n_in = data2.shape[1]
#make the weights
w2 = np.random.randn(number_nodes, n_in) * 0.1
#do the training and show the weights on the nodes with cv2
for i in range(n_iters):
    randsamples = np.random.randint(0, data2.shape[0], 1)[0] 
    rand_in = data2[randsamples, :] 
    difference = w2 - rand_in
    dist = np.sum(np.absolute(difference), 1)
    best = np.argmin(dist)
    w2_eligible = w2[best,:]
    w2_eligible += (lr * (rand_in - w2_eligible))
    w2[best,:] = w2_eligible
    cv2.namedWindow('weights', cv2.WINDOW_NORMAL)
    cv2.imshow('weights', bytescale(w2))
    cv2.waitKey(100)
#queary validation   
node4 = w2[0, :]
node5 = w2[1, :]
node6 = w2[2, :]
difference4 = node4 - data
dist4 = np.sum(np.abs(difference4), 1)
difference5 = node5 - data
dist5 = np.sum(np.abs(difference5), 1)
difference6 = node6 - data
dist6 = np.sum(np.abs(difference6), 1)
top4 = np.argmin(dist4)
top5 = np.argmin(dist5)
top6 = np.argmin(dist6)
ans4 = namedata[top4]
ans5 = namedata[top5]
ans6 = namedata[top6]
ans2 = []
ans2.append(ans4)
ans2.append(ans5)
ans2.append(ans6)

#show the weights for the datafile
print(filename2)
print(w2)
print(ans2)
###############################################################################

