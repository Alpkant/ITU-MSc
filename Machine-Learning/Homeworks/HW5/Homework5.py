import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import math, random
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('data.txt', delimiter=",", header = None)
data = data.to_numpy()
# Randomize the input
np.random.shuffle(data)
data_features = data[:,:64]
min_max_scaler = MinMaxScaler((-1 ,1))
data_features = min_max_scaler.fit_transform(data_features)
data_labels = data[:,-1:]

def sigmoid(x, derive=False):
    if derive:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    ex = np.exp(-x)
    y = ex / (1 + ex)**2
    return y

learning_rate = 0.7
epochs = 1000
w1 = np.random.uniform(0, 1, (64, 2))
w2 = np.random.uniform(0, 1, (2, 64))

b1 = np.full((1, 2), 0.1)
b2 = np.full((1, 64), 0.1)

for i in range(epochs):
    a1 = data_features.copy()
    z2 = a1.dot(w1) + b1
    a2 = sigmoid(z2)
    z3 = a2.dot(w2) + b2
    a3 = sigmoid(z3)

    cost = np.sum((a3 - data_features)**2)/2
    
    a3_delta = (a3-data_features)
    z3_delta = sigmoid_derivative(a3_delta)
    dw2 = a2.T.dot(z3_delta)
    db2 = np.sum(z3_delta,axis=0, keepdims=True)
    
    z2_delta = z3_delta.dot(w2.T) * sigmoid_derivative(z2)
    dw1 = a1.T.dot(z2_delta)
    db1 = np.sum(z2_delta,axis=0, keepdims=True)

    # update parameters
    for param, gradient in zip([w1, w2, b1, b2], [dw1, dw2, db1, db2]):
        param -= learning_rate * gradient/(gradient*gradient)
    if i %100 == 0:
        print("Epoch: {}/{} Error: {:.4f}".format(i,epochs,cost))

inputs = data_features
before_active = inputs.dot(w1) + b1
predictions = sigmoid(before_active)

plt.figure()
colors = ['red','green','blue','purple','black',"yellow","orange","cyan","magenta"]


samples = random.sample(range(0,len(predictions[:,0])),5)   #Take random 100 points
plt.scatter(predictions[:,0],predictions[:,1])
for i in samples:
    
    plt.annotate(data_labels[i,0],(predictions[i,0],predictions[i,1]),size=16) #Draw classes of the points

plt.show()

