####################################################################################################
## A simple feed forward network using tensorflow and some of its visualization tools
##Architecture
## 2 hidden layers 1 input and 1 output layers
## input layer : 10 neurons corresponding to season, mnth,holiday,weekday,workingday, weathersit, temp, atemp, hum, windspeed
##hidden layers with 5 and 3 neurons respectively
##output neuron. This is a regression type of problem where the output value predicts the answer "cnt" in the dataset.
####################################################################################################

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from matplotlib import pyplot as plt

#preprocessing the data
path="day.csv"
dataset=pd.read_csv(path)
costHistory=[]
learningRate=0.5
totalepoch=3000
samplesize=90
dataset=dataset.drop(['instant','dteday','casual','registered','yr'],axis=1)

#factors being used are season, mnth,holiday,workingday, weathersit, temp, atemp, hum, windspeed, cnt
dataset=shuffle(dataset)
####create tensor graph

#create placeholder to inject input to the tensorgraph
X=tf.placeholder(dtype="float",shape=[None,10],name="x-input")
Y=tf.placeholder(dtype="float",shape=[None,1],name='output')
weights={'w1':tf.Variable(tf.random_uniform([10,5],minval=1,maxval=9)),
         'w2':tf.Variable(tf.random_uniform([5,1],minval=1,maxval=9))} #weights and biases as a dictionary
biases={'b1':tf.Variable(tf.constant(0.5)),
        'b2':tf.Variable(tf.constant(0.3))}

layer1_output=tf.nn.relu6(tf.matmul(X,weights['w1']))
layer2_output=tf.nn.sigmoid(tf.matmul(layer1_output,weights['w2']))

cost=tf.reduce_sum(tf.pow((Y-layer2_output),1),axis=1)
optimizer=tf.train.GradientDescentOptimizer(learning_rate=learningRate).minimize(cost)

#run the graph
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(0,totalepoch):
        trainingSample = dataset.sample(samplesize)
        cnt = np.asarray(trainingSample['cnt']).reshape([samplesize,1])
        trainingSample.drop(['cnt'], axis=1)
        inparray=np.asarray([trainingSample['season'],trainingSample['mnth'],trainingSample['holiday'],trainingSample['weekday'],trainingSample['workingday'],trainingSample['weathersit'],trainingSample['temp'],trainingSample['atemp'],trainingSample['hum'],trainingSample['windspeed']])
        inparray=inparray.transpose()
        #print(inparray.shape)
        #print(cnt.shape)
        sess.run(optimizer,feed_dict={X:inparray,Y:cnt})
        cst =sess.run(cost,feed_dict={X:inparray,Y:cnt})
        costHistory.append(cst)

    plt.plot(range(len(costHistory)), costHistory)
    plt.show()
