#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 18:21:56 2018

@author: mpcr
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from scipy.misc import bytescale


#define hyperparameters
lr = 0.025
n_iters = 100
number_nodes = 3
#get the data for all sheets independently
ws = []
file = ['DE0', 'DE1', 'DE2', 'DE3']
for DE in file:
    filename = DE + '.csv'
    d = np.genfromtxt(filename, delimiter=',', missing_values='NA', skip_header=0, filling_values=1, usecols=range(1,7))
    removeNA = d[:, -1] != 1
    d = d[removeNA, :]
    print(d.shape)
#get the data with the gene names in first column as a string
    nd = np.genfromtxt(filename, delimiter=',', skip_header=0,  dtype=str, usecols=0)
    nd = nd[removeNA]
    nd = nd[:] 
    print(nd.shape)
#normalize the data    
    d = np.array(d, dtype=np.float64)
    d -= np.mean(d, 0)
    d /= np.std(d, 0)
    print(d.shape)
#make the weights
    n_in = d.shape[1]
    w = np.random.randn(number_nodes, n_in) * 0.1
#do the training and show the weights on the nodes with cv2
    for i in range(n_iters):
        randsamples = np.random.randint(0, d.shape[0], 1)[0] 
        rand_in = d[randsamples, :] 
        difference = w - rand_in
        dist = np.sum(np.absolute(difference), 1)
        best = np.argmin(dist)
        w_eligible = w[best,:]
        w_eligible += (lr * (rand_in - w_eligible))
        w[best,:] = w_eligible
        cv2.namedWindow('weights', cv2.WINDOW_NORMAL)
        cv2.imshow('weights', bytescale(w))
        cv2.waitKey(100)
#save weights
    ws.append(w)
print(ws)
#seperate weights per file
w0 = ws[0]
w1 = ws[1]
w2 = ws[2]
w3 = ws[3]

#query validation  
down = []
up = []
neut = []
for w in ws:
    logcol = w[:, 1]
    downcol = np.argmin(logcol)
    upcol = np.argmax(logcol)      
    node1 = w[downcol, :]
    node2 = w[upcol, :]
    node3 = np.delete(w, [downcol, upcol], axis=0)               
    print(node1)
    
#get the best fit and save it in down/up/neut
for 
    difference1 = node1 - d
    dist1 = np.sum(np.abs(difference1), 1)
    top1 = np.argmin(dist1)
    ans1 = nd[top1]
    print(ans1)
    down.insert(0, ans1) 
    difference2 = node2 - d
    dist2 = np.sum(np.abs(difference2), 1)
    top2 = np.argmin(dist2)
    ans2 = nd[top2]
    up.insert(0, ans2)
    difference3 = node3 - d
    dist3 = np.sum(np.abs(difference3), 1)
    top3 = np.argmin(dist3)
    ans3 = nd[top3]
    neut.insert(0, ans3)

#validation query
def nodes():
    nodes = []
    for i in range(0, number_nodes):
        nodename = 'node'
        number = i
        nodename = nodename + str(number)
        nodes.append(nodename)
    nodes = np.asarray(nodes)
    nodes.shape = (number_nodes, 1)
    nnodes = []
    for node in nodes[:, 2]:
        if node < 0.5:
            nodetype = 'downregulated'
            nodes = nodes.append(nodetype, axis=1)
        if node > 0.5:
            nodetype = 'upregulated'
            nodes = nodes.append(nodetype, axis=1)
        else:
            nodetype = 'neutral' 
            nodes = nodes.append(nodetype, axis=1)
    for nodename in nodes:
        nodeval = n[number_nodes, :]
        nnodes.append(nodeval)
    nnodes = np.asarray(nnodes)
    nnodes.shape = (number_nodes, n_in)
    nodes = nodes.append(nnodes, axis=1)
    print(nodes)
nodes()
    
def nodetype (nodes):
    for node in nodes[:, 2]:
        if node < 0.5:
            nodetype = 'downregulated'
        if node > 0.5:
            nodetype = 'upregulated'
        else:
            nodetype = 'neutral'             

def diff():
    differences = []
    for nodename in nnodes:
        difference = 'diff' + str(number)
        differences.append(difference)
def difference (nodes):
    for difference in differences:
        diff = nodename - data
    print(differences)
difference(diff2)

   
    
   
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
print (filename)
print (w)
print(ans)
#make the weights
w2 = np.random.randn(number_nodes, n_in) * 0.1
#do the training and show the weights on the nodes with cv2
for i in range(n_iters):
    randsamples = np.random.randint(0, data2.shape[0], 1)[0] #between zero and number of rows, give me one
    rand_in = data2[randsamples, :] #use randsample as an index for that row and all columns associated
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
dist4 = dist4[dist4 != np.amax(dist4)]
dist5 = dist5[dist5 != np.amax(dist5)]
dist6 = dist6[dist6 != np.amax(dist6)]
dist4 = dist4[dist4 != np.amax(dist4)]
dist5 = dist5[dist5 != np.amax(dist5)]
dist6 = dist6[dist6 != np.amax(dist6)]  
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
###plot the data in 3d topology graph
fig = plt.figure()
ax= fig.add_subplot(111, projection='3d')
ax.scatter(dist4, dist5, dist6)
for x in range(360):
    ax.view_init(elev=90, azim=x)  # can change the view like this...elev = viewing elevation; azim = viewing rotation
    plt.tight_layout()
    plt.show()
    plt.close()
#show the weights for the datafile
print(filename2)
print(w2)
print(ans2)
###############################################################################

