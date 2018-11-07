#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 12:44:33 2018
@author: mpcr
"""

##self-organizing map##
#DESeq2 data for MMP9 t-cell inhibititor conditions#

import numpy as np
import csv
import collections, numpy
import matplotlib.pyplot as plt

#how many times to run the whole thing
n_trials = 10
#define hyperparameters
lr = 0.001
n_iters = 1000
number_nodes = 3
#store the answers here
downhigh = []
downlow = []
uphigh = []
uplow = []
neuthigh = []
neutlow = []
D = []
#get the data for all sheets independently and mkae weight
file = ['DE0', 'DE1', 'DE2', 'DE3']
for DE in file:
    filename = DE + '.csv'
    d = np.genfromtxt(filename, delimiter=',', missing_values='NA', skip_header=0, filling_values=1, usecols=range(1,7))
    removeNA = d[:, -1] != 1
    d = d[removeNA, :]
    #get the data with the gene names in first column as a string
    nd = np.genfromtxt(filename, delimiter=',', skip_header=0,  dtype=str, usecols=0)
    nd = nd[removeNA]
    nd = nd[:] 
    #nomalize the data    
    d = np.array(d, dtype=np.float64)
    d -= np.mean(d, 0)
    d /= np.std(d, 0)
    D.append(d)
d0 = D[0]
d1 = D[1]
d2 = D[2]
d2 = np.delete(d2, 7861, axis=0)
d3 = D[3]                  
def trial(x):
    for instance in range(n_trials):
        ws = [] #weights per sheet
        for d in D:
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
            ws.append(w)
        #seperate the data per file
        N1 = []
        N2 = []
        N3 = []
        #seperate weights per file
        w0 = ws[0]
        w1 = ws[1]
        w2 = ws[2]
        w3 = ws[3]
        #specify type of weights  
        for w in ws:
            logcol = w[:, 1]
            downcol = np.argmin(logcol)
            upcol = np.argmax(logcol)      
            node1 = w[downcol, :]
            node2 = w[upcol, :]
            node3 = np.delete(w, [downcol, upcol], axis=0)               
            N1.append(node1)
            N2.append(node2)
            N3.append(node3)  
        #get the best fit and save it in down/up/neut
        for i in range(0, 4):
            #downregulated
            difference1 = N1[i] - x
            dist1 = np.sum(np.abs(difference1), 1)        
            top1 = np.argmin(dist1)
            bot1 = np.argmax(dist1)
            low1 = nd[bot1]
            high1 = nd[top1]
            downhigh.insert(0, high1) 
            downlow.insert(0, low1)
            #upregulated
            difference2 = N2[i] - x
            dist2 = np.sum(np.abs(difference2), 1)
            top2 = np.argmin(dist2)
            bot2 = np.argmax(dist2)
            low2 = nd[bot2]
            high2 = nd[top2]
            uphigh.insert(0, high2)
            uplow.insert(0, low2)
            #neutral 
            difference3 = N3[i] - x
            dist3 = np.sum(np.abs(difference3), 1)
            top3 = np.argmin(dist3)
            bot3 = np.argmax(dist3)
            low3 = nd[bot3]
            high3 = nd[top3]
            neuthigh.insert(0, high3)
            neutlow.insert(0, low3)
#run it########################################################################
trial(d2)
#COUNTER#######################################################################
##format the regualted gene names into an array
#col1=upregulated; col2=downregulated
downr = np.stack((downhigh, downlow), axis=1)
upr = np.stack((uphigh, downlow), axis=1)
neutr = np.stack((neuthigh, neutlow), axis=1)
#each set of 4k rows is down, up and neut respectively
bothr = np.concatenate((downr, upr), axis=0)
regulated = np.concatenate((bothr, neutr), axis=0)
print(regulated.shape)
regulated = tuple(map(tuple, regulated))
#find the top ten occuring names in each set for high, low
top_reg = []
for cond in range(3):
    num = n_trials*4
    batch = (cond*num) + num
    insert = regulated[:batch]
    insert = tuple(map(tuple, insert))
    count = collections.Counter(insert)
    count = np.asarray(count.most_common())
    topten = count[:10]
    top_reg.append(topten)
print(top_reg)
#save the top10 regulated from (down, up, and neut) of (high, low) as .csv#####

#csvname = 'd2' + str(n_trials) + '_top_reg'
#csvfile = csvname + '.csv'
#with open(csvfile, mode='w', newline='') as csvname:
#    gene_writer = csv.writer(csvname, delimiter=',')
#    gene_writer.writerow(top_reg)


        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    