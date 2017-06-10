import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.utils import shuffle


#####Read the data and preprocess it
##The dataset is taken from uci machine learning repository(http://archive.ics.uci.edu/ml/)
##the details about the dataset is in readme.txt
path="day.csv"
dataset=pd.read_csv(path)
costHistory=[]
dataset=dataset.drop(['instant','dteday','casual','registered','yr'],axis=1)

#factors being used are season, mnth,holiday,workingday, weathersit, temp, atemp, hum, windspeed, cnt
#shuffling the dataset
dataset=shuffle(dataset)
dataset=shuffle(dataset)
learningrate=0.1
trainingEpoch=1000
displayStep=50
rng=np.random
#out of 730 data instances, i have taken 600 of them for training and remaining 130 for validation
#validation_data
Valx1=dataset['season'][601:]
Valx2=dataset['mnth'][601:]
Valx3=dataset['holiday'][601:]
Valx33=dataset['weekday'][601:]
Valx4=dataset['workingday'][601:]
Valx5=dataset['weathersit'][601:]
Valx6=dataset['temp'][601:]
Valx7=dataset['atemp'][601:]
Valx8=dataset['hum'][601:]
Valx9=dataset['windspeed'][601:]
ValYY=dataset['cnt'][601:]
nofValSamples=ValYY.shape[0]
#training_data
trainx1=dataset['season'][:600]
trainx2=dataset['mnth'][:600]
trainx3=dataset['holiday'][:600]
trainx33=dataset['weekday'][:600]
trainx4=dataset['workingday'][:600]
trainx5=dataset['weathersit'][:600]
trainx6=dataset['temp'][:600]
trainx7=dataset['atemp'][:600]
trainx8=dataset['hum'][:600]
trainx9=dataset['windspeed'][:600]
trainYY=dataset['cnt'][:600]
noftrainSamples=trainYY.shape[0]
print(nofValSamples)
print(noftrainSamples)

###Build the graph
w1=tf.Variable(rng.random(),'w1Season')
w2=tf.Variable(rng.random(),'w2mnth')
w3=tf.Variable(rng.random(),'w3holiday')
w33=tf.Variable(rng.random(),'w33weekday')
w4=tf.Variable(rng.random(),'w4workingday')
w5=tf.Variable(rng.random(),'w5weathersit')
w6=tf.Variable(rng.rand(),'w6temp')
w7=tf.Variable(rng.random(),'w7atemp')
w8=tf.Variable(rng.random(),'w8hum')
w9=tf.Variable(rng.random(),'w9windspeed')
b=tf.Variable(rng.randn(),'bias')

Y=tf.placeholder("float")

season=tf.placeholder("float")
mnth=tf.placeholder("float")
holiday=tf.placeholder("float")
weekday=tf.placeholder("float")
workingday=tf.placeholder("float")
weathersit=tf.placeholder("float")
temp=tf.placeholder("float")
atemp=tf.placeholder("float")
hum=tf.placeholder("float")
windspeed=tf.placeholder("float")

#Y=b+w1*Season+w2*mnth+w3*holiday+w4*workingday+w5*weathersit+w6*temp+w7*atemp+w8*hum+w9*windspeed

pred=tf.add(tf.multiply(season,w1),tf.add(tf.multiply(mnth,w2),tf.add(tf.multiply(holiday,w3),tf.add(tf.multiply(workingday,w4),tf.add(tf.multiply(weathersit,w5),tf.add(tf.multiply(temp,w6),tf.add(tf.multiply(atemp,w7),tf.add(tf.multiply(hum,w8),tf.add(tf.multiply(weekday,w33),tf.multiply(windspeed,w9))))))))))
predfinal=tf.add(pred,b)
cost=tf.reduce_sum(tf.pow(predfinal-Y,2))/noftrainSamples
optimizer=tf.train.GradientDescentOptimizer(learning_rate=learningrate).minimize(cost)


trainYY=np.reshape(trainYY,(trainYY.shape[0],1))

ValYY=np.reshape(ValYY,(ValYY.shape[0],1))

init=tf.global_variables_initializer()

###train the weights

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(trainingEpoch):
        for(s,m,h,wee,w,ws,t,at,h,ws,y) in zip(trainx1,trainx2,trainx3,trainx33,trainx4,trainx5,trainx6,trainx7,trainx8,trainx9,trainYY):

            sess.run(optimizer,feed_dict={season:s,mnth:m,holiday:h,weekday:wee,workingday:w,weathersit:ws,temp:t,atemp:at,hum:h,windspeed:ws,Y:y})
        if(epoch + 1)%displayStep ==0:
            csmt=sess.run(cost,feed_dict={season:s,mnth:m,holiday:h,weekday:wee,workingday:w,weathersit:ws,temp:t,atemp:at,hum:h,windspeed:ws,Y:y})
            costHistory.append(csmt)
            print('cost '+str(csmt)+' _w1: '+str(sess.run(w1))+' _w2: '+str(sess.run(w2))+' _w3: '+str(sess.run(w3))+' _w33: '+str(sess.run(w33))+' _w4: '+str(sess.run(w4))+' _w5: '+str(sess.run(w5))+' _w6: '+str(sess.run(w6))+' _w7: '+str(sess.run(w7))+' _w8: '+str(sess.run(w8))+' _w9: '+str(sess.run(w9)))


    print("training completed.............................Running Validation...")
    cst = sess.run(cost,
                   feed_dict={season: Valx1, mnth: Valx2, holiday: Valx3, weekday: Valx33, workingday: Valx4,
                              weathersit: Valx5,
                              temp: Valx6, atemp: Valx7, hum: Valx8, windspeed: Valx9, Y: ValYY})
    print("cost during validation: "+str(cst))
    plt.plot(range(len(costHistory)),costHistory)
    plt.show()
