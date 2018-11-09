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
import operator
import matplotlib.pyplot as plt

#how many times to run the whole thing
n_trials = 1
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
ND = []
#get the data for all sheets independently and mkae weight#####################
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
    ND.append(nd)
    #nomalize the data    
    d = np.array(d, dtype=np.float64)
    d -= np.mean(d, 0)
    d /= np.std(d, 0)
    D.append(d)
#view the data for exceptions##################################################
d0 = D[0]
d1 = D[1]
d2 = D[2]
d3 = D[3]
nd0 = ND[0]
nd1 = ND[1]
nd2 = ND[2]
nd3 = ND[3]  
#do the training###############################################################             
def trial(x, y):
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
        #get the best fit and save it in down/up/neut for each node############
        for i in range(0, 4):# range i == each datasheet weight of N#
        #downregulated
            d0 = D[0]
            d1 = D[1]
            d2 = D[2]
            d3 = D[3]
            nd0 = ND[0]
            nd1 = ND[1]
            nd2 = ND[2]
            nd3 = ND[3]  
            outliershigh = []
            for r in range(20):#range 20 = take the top 20 outliers (argmax/min)
                difference1 = N1[i] - x
                dist1 = np.sum(np.abs(difference1), 1)
                top1 = np.argmin(dist1)
                outliershigh.insert(-1, top1)
                x = np.delete(x, top1, axis=0)
            olhigh = []
            for z in range(20):
                ind = outliershigh[z]
                thing1 = y[ind]
                y = np.delete(y, ind, axis=0)
                olhigh.insert(-1, thing1)
            downhigh.insert(0, olhigh)
            d0 = D[0]
            d1 = D[1]
            d2 = D[2]
            d3 = D[3]
            nd0 = ND[0]
            nd1 = ND[1]
            nd2 = ND[2]
            nd3 = ND[3]  
            outlierslow = []
            for r in range(20):
                difference1 = N1[i] - x
                dist1 = np.sum(np.abs(difference1), 1)
                bot1 = np.argmax(dist1)
                outlierslow.insert(-1, bot1)
                x = np.delete(x, bot1, axis=0)
            ollow = []
            for z in range(20):
                ind = outlierslow[z]
                thing2 = y[ind]
                y = np.delete(y, ind, axis=0)
                ollow.insert(-1, thing2)
            downlow.insert(0, ollow)   
        #upregulated#####################################################
            d0 = D[0]
            d1 = D[1]
            d2 = D[2]
            d3 = D[3]
            nd0 = ND[0]
            nd1 = ND[1]
            nd2 = ND[2]
            nd3 = ND[3]  
            outliershigh = []
            for r in range(20):
                difference2 = N2[i] - x
                dist2 = np.sum(np.abs(difference2), 1)
                top1 = np.argmin(dist2)
                outliershigh.insert(-1, top1)
                x = np.delete(x, top1, axis=0)
            olhigh = []
            for z in range(20):
                ind = outliershigh[z]
                thing1 = y[ind]
                y = np.delete(y, ind, axis=0)
                olhigh.insert(-1, thing1)
            uphigh.insert(0, olhigh)
            d0 = D[0]
            d1 = D[1]
            d2 = D[2]
            d3 = D[3]
            nd0 = ND[0]
            nd1 = ND[1]
            nd2 = ND[2]
            nd3 = ND[3]  
            outlierslow = []
            for r in range(20):
                difference2 = N2[i] - x
                dist2 = np.sum(np.abs(difference2), 1)
                bot1 = np.argmax(dist2)
                outlierslow.insert(-1, bot1)
                x = np.delete(x, bot1, axis=0)
            ollow = []
            for zl in range(20):
                ind = outlierslow[zl]
                thing2 = y[ind]
                y = np.delete(y, ind, axis=0)
                ollow.insert(-1, thing2)
            uplow.insert(0, ollow) 
        #neutral########################################################
            d0 = D[0]
            d1 = D[1]
            d2 = D[2]
            d3 = D[3]
            nd0 = ND[0]
            nd1 = ND[1]
            nd2 = ND[2]
            nd3 = ND[3]  
            outliershigh = []
            for r in range(20):
                difference3 = N3[i] - x
                dist3 = np.sum(np.abs(difference3), 1)
                top1 = np.argmin(dist3)
                outliershigh.insert(-1, top1)
                x = np.delete(x, top1, axis=0)
            olhigh = []
            for z in range(20):
                ind = outliershigh[z]
                thing1 = y[ind]
                y = np.delete(y, ind, axis=0)
                olhigh.insert(-1, thing1)
            neuthigh.insert(0, olhigh)
            d0 = D[0]
            d1 = D[1]
            d2 = D[2]
            d3 = D[3]
            nd0 = ND[0]
            nd1 = ND[1]
            nd2 = ND[2]
            nd3 = ND[3]  
            outlierslow = []
            for r in range(20):
                difference3 = N3[i] - x
                dist3 = np.sum(np.abs(difference3), 1)
                bot1 = np.argmax(dist3)
                outlierslow.insert(-1, bot1)
                x = np.delete(x, bot1, axis=0)
            ollow = []
            for zl in range(20):
                ind = outlierslow[zl]
                thing2 = y[ind]
                y = np.delete(y, ind, axis=0)
                ollow.insert(-1, thing2)
            neutlow.insert(0, ollow) 
#run it########################################################################
trial(d2, nd2)
#COUNTER#######################################################################
#sets
downhigh = np.asarray(downhigh),
downlow = np.asarray(downlow),
uphigh = np.asarray(uphigh),
uplow = np.asarray(uplow),
neuthigh = np.asarray(neuthigh),
neutlow = np.asarray(neutlow)
sets = downhigh, downlow, uphigh, uplow, neuthigh, neutlow
#find the top ten occuring names in each set for high, low
#?????????the problem is if there is not the highest??????????????????????????????
each = []
for element in sets:
    top_reg = []
    for cond in range(4):#cond=row of downhigh = 
        count = np.array(np.unique(element,return_counts=1)).transpose()
        counted = np.asarray(count)
        #count.argsort()
        index, value = max(enumerate(count[1]), key=operator.itemgetter(1))
        top_ten = counted[0, index] 
        top_reg.append(top_ten)
    each.append(top_reg)
    print(top_reg)



#save the top10 regulated from (down, up, and neut) of (high, low) as .csv#####
#csvname = 'd2' + str(n_trials) + '_top_reg'
#csvfile = csvname + '.csv'
#with open(csvfile, mode='w', newline='') as csvname:
#    gene_writer = csv.writer(csvname, delimiter=',')
#    gene_writer.writerow(top_reg)

#for cond in range(3):
#    num = n_trials*4
#    batch = (cond*num) + num
#    insert = regulated[:batch] 
    #count = collections.Counter(insert)
    #count = np.asarray(count.most_common())
#    topten = count[:10]
#    top_reg.append(topten)
#print(top_reg)
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    