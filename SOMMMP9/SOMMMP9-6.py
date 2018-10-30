#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 09:42:06 2018

@author: mpcr
"""


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from scipy.misc import bytescale
from mpl_toolkits.mplot3d import Axes3D
import time
import csv
import collections, numpy


down = []
up = []
neut = []
file = 'DE0'
filename = file + '.csv'
data = np.genfromtxt(filename, delimiter=',', missing_values='NA', filling_values=1, skip_header=2, usecols=range(1,7))
removeNA = data[:, -1] != 1
data = data[removeNA, :]
print(data.shape)
#get the data with the gene names in first column as a string
namedata = np.genfromtxt(filename, delimiter=',', dtype=str, skip_header=2, usecols=0)
namedata = namedata[removeNA]
namedata = namedata[:] 
print(namedata.shape)
#convert the string gene names to integers and put them in a new dataset 
dataname = []
for name in namedata:
    input = name
    input = input.lower()
    numbername = []
    for character in input:
        number = ord(character) 
        numbername.append(number)
    dataname.append(numbername)
len(dataname)
#make the names a single integer instead of comma seperated integers
finalnames = []
for element in dataname:
    numbername = int("".join(map(str, element)))
    type(numbername)
    finalnames.append(numbername)
len(finalnames)   
namesarray = np.asarray(finalnames)
namesarray.shape = (7903, 1)
print(namesarray.shape)
#attach single integer names to their numerical properties (columns1-6)
data = np.concatenate((namesarray, data), axis=1)
print(data.shape)
print(data[0])
#normalize the data
data = np.array(data, dtype=np.float64)
print(data.shape)
data -= np.mean(data, 0)
print(data.shape)
data /= np.std(data, 0)
print(data.shape)
#create an array that has the condition
condition = np.full((7903, 1), 0)
print(condition.shape)
#add the condition array to the data
data = np.concatenate((data, condition), axis=1)
print(data[0])
print(data.shape)
###############################################################################
file1 = 'DE1'
filename1 = file1 + '.csv'
data1 = np.genfromtxt(filename1, delimiter=',', missing_values='NA', filling_values=1, skip_header=2, usecols=range(1,7))
removeNA1 = data1[:, -1] != 1
data1 = data1[removeNA1, :]
print(data1.shape)
#get the data with the gene names in first column as a string
namedata1 = np.genfromtxt(filename1, delimiter=',', dtype=str, skip_header=2, usecols=0)
namedata1 = namedata1[removeNA1]
namedata1 = namedata1[:] 
print(namedata1.shape)
#convert the string gene names to integers and put them in a new dataset 
dataname1 = []
for name1 in namedata1:
    input = name1
    input = input.lower()
    numbername1 = []
    for character in input:
        number = ord(character) 
        numbername1.append(number)
    dataname1.append(numbername1)
len(dataname1)
#make the names a single integer instead of comma seperated integers
finalnames1 = []
for element in dataname1:
    numbername1 = int("".join(map(str, element)))
    type(numbername1)
    finalnames1.append(numbername1)
len(finalnames1)   
namesarray1 = np.asarray(finalnames1)
namesarray1.shape = (5815, 1)
print(namesarray1.shape)
#attach single integer names to their numerical properties (columns1-6)
data1 = np.concatenate((namesarray1, data1), axis=1)
print(data1.shape)
print(data1[0])
#normalize the data
data1 = np.array(data1, dtype=np.float64)
print(data1.shape)
data1 -= np.mean(data1, 0)
print(data1.shape)
data1 /= np.std(data1, 0)
print(data1.shape)
#create an array that has the condition
condition1 = np.full((5815, 1), 1)
print(condition1.shape)
#add the condition array to the data
data1 = np.concatenate((data1, condition1), axis=1)
print(data1[0])
print(data1.shape)
###############################################################################
file2 = 'DE2'
filename2 = file2 + '.csv'
data2 = np.genfromtxt(filename2, delimiter=',', missing_values='NA', filling_values=1, skip_header=2, usecols=range(1,7))
removeNA2 = data2[:, -1] != 1
data2 = data2[removeNA2, :]
print(data2.shape)
#get the data with the gene names in first column as a string
namedata2 = np.genfromtxt(filename2, delimiter=',', dtype=str, skip_header=2, usecols=0)
namedata2 = namedata2[removeNA2]
namedata2 = namedata2[:] 
print(namedata2.shape)
#convert the string gene names to integers and put them in a new dataset 
dataname2 = []
for name in namedata2:
    input = name
    input = input.lower()
    numbername2 = []
    for character in input:
        number = ord(character) 
        numbername2.append(number)
    dataname2.append(numbername2)
len(dataname2)
#make the names a single integer instead of comma seperated integers
finalnames2 = []
for element in dataname2:
    numbername2 = int("".join(map(str, element)))
    type(numbername2)
    finalnames2.append(numbername2)
len(finalnames2)   
namesarray2 = np.asarray(finalnames2)
namesarray2.shape = (10848, 1)
print(namesarray2.shape)
#attach single integer names to their numerical properties (columns1-6)
data2 = np.concatenate((namesarray2, data2), axis=1)
print(data2.shape)
print(data2[0])
#normalize the data
data2 = np.array(data2, dtype=np.float64)
print(data2.shape)
data2 -= np.mean(data2, 0)
print(data2.shape)
data2 /= np.std(data2, 0)
print(data2.shape)
#create an array that has the condition
condition2 = np.full((10848, 1), 2)
print(condition2.shape)
#add the condition array to the data
data2 = np.concatenate((data2, condition2), axis=1)
print(data2[0])
print(data2.shape)
###############################################################################
file3 = 'DE3'
filename3 = file3 + '.csv'
data3 = np.genfromtxt(filename3, delimiter=',', missing_values='NA', filling_values=1, skip_header=2, usecols=range(1,7))
removeNA3 = data3[:, -1] != 1
data3 = data3[removeNA3, :]
print(data3.shape)
#get the data with the gene names in first column as a string
namedata3 = np.genfromtxt(filename3, delimiter=',', dtype=str, skip_header=2, usecols=0)
namedata3 = namedata3[removeNA3]
namedata3 = namedata3[:] 
print(namedata3.shape)
#convert the string gene names to integers and put them in a new dataset 
dataname3 = []
for name in namedata3:
    input = name
    input = input.lower()
    numbername3 = []
    for character in input:
        number = ord(character) 
        numbername3.append(number)
    dataname3.append(numbername3)
len(dataname3)
#make the names a single integer instead of comma seperated integers
finalnames3 = []
for element in dataname3:
    numbername3 = int("".join(map(str, element)))
    type(numbername3)
    finalnames3.append(numbername3)
len(finalnames3)   
namesarray3 = np.asarray(finalnames3)
namesarray3.shape = (11862, 1)
print(namesarray3.shape)
#attach single integer names to their numerical properties (columns1-6)
data3 = np.concatenate((namesarray3, data3), axis=1)
print(data3.shape)
print(data3[0])
#normalize the data
data3 = np.array(data3, dtype=np.float64)
print(data3.shape)
data3 -= np.mean(data3, 0)
print(data3.shape)
data3 /= np.std(data3, 0)
print(data3.shape)
#create an array that has the condition
condition3 = np.full((11862, 1), 3)
print(condition3.shape)
#add the condition array to the data
data3 = np.concatenate((data3, condition3), axis=1)
print(data3[0])
print(data3.shape)
###############################################################################
#put all the data in one array
data = np.vstack((data, data1))
print(data.shape)
data = np.vstack((data, data2))
print(data.shape)
data = np.vstack((data, data3))
print(data[0])
print(data.shape)
###############################################################################
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
n_iters = 1000
#do the training and show the weights on the nodes with cv2
def trial(n_trial):
    for instance in range(n_trial): 
        #specify the columns to later be represented by nodes
        n_in = data.shape[1]
        #make the weights
        number_nodes = 3
        w = np.random.randn(number_nodes, n_in) * 0.1
        #hyperparameters
        lr = 0.025
        n_iters = 500
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
            #cv2.imshow('weights', bytescale(w))
            #cv2.waitKey(100)
        #cv2.destroyAllWindows()
#query validation   
        logcol = w[:, 1]
        downcol = np.argmin(logcol)
        upcol = np.argmax(logcol)
        
        node1 = w[downcol, :]
        node2 = w[upcol, :]
        node3 = np.delete(w, [downcol, upcol], axis=0)               

        difference1 = node1 - data
        dist1 = np.sum(np.abs(difference1), 1)
        top1 = np.argmin(dist1)
        ans1 = namedata[top1]
        print(ans1)
        down.insert(0, ans1) 
        difference2 = node2 - data
        dist2 = np.sum(np.abs(difference2), 1)
        top2 = np.argmin(dist2)
        ans2 = namedata[top2]
        up.insert(0, ans2)
        difference3 = node3 - data
        dist3 = np.sum(np.abs(difference3), 1)
        top3 = np.argmin(dist3)
        ans3 = namedata[top3]
        neut.insert(0, ans3)
#show the results for the query validation
    print (filename)
    print(node1, node2, node3)
    print(down[0], up[0], neut[0])
n_trial = 3
trial(n_trial)
csvfile = 'genes' + str(n_trial) + filename
csvname = 'genes' + str(n_trial) + file
with open(csvfile, mode='w', newline='') as csvname:
    gene_writer = csv.writer(csvname, delimiter=',')
    gene_writer.writerow(down)
    gene_writer.writerow(up)
    gene_writer.writerow(neut)
###############################################################################

genes = np.genfromtxt(csvfile, delimiter=',', dtype=str)
print(genes.shape)
downgenes = genes[0, :]
print(downgenes.shape)
upgenes = genes[1, :]
neutgenes = genes[2, :]
print(neutgenes.shape)
downv = collections.Counter(downgenes)
downv = np.asarray(downv.most_common())
print(downv.shape)    

upv = collections.Counter(upgenes)
upv = np.asarray(upv.most_common())
print(upv.shape) 

neutv = collections.Counter(neutgenes)
neutv = np.asarray(neutv.most_common())
print(neutv.shape) 

#downplot = plt.bar(down[:, 0], down[:, 1])
#upplot = plt.hist(up)
#neutplot = plt.hist(neut)
#plt.show(downplot)
#plt.show(upplot)
#print(down, up, neut)

both = []
for item in downv[:, 0]:
    if item in upv[:, 0]:
        both.insert(0, item)
print(both)

csvfile1 = file + 'geneanalysis.csv'
csvname1 = file + 'geneanalysis'
with open(csvfile1, mode='w', newline='') as csvname1:
    gene_writer = csv.writer(csvname1, delimiter=',')
    gene_writer.writerow(downv)
    gene_writer.writerow(upv)
    gene_writer.writerow(neutv)