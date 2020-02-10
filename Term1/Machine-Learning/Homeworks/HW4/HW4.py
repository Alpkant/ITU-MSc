#!/usr/bin/env python
# coding: utf-8

# In[1]:


# BLG527E Machine Learning Homework4
# Author: Alperen Kantarcı
# No: 504191504

import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
from matplotlib.colors import ListedColormap
import random


# In[2]:


# Get the data points
dataset = loadmat('q2.mat')['d']
# Shuffle dataset for better cross validation learning
np.random.shuffle(dataset)


# In[3]:


num_points = dataset.shape[0]
# Add full of ones that represents x0 feature
ones = np.full((num_points,1),1)
dataset = np.hstack((ones,dataset))
num_features = len(dataset[0,:-1])


# In[4]:


one_class = []
zero_class = []
for i in range(num_points):
    if dataset[i,-1] == 0:
        zero_class.append(dataset[i,1:-1])
    else:
        one_class.append(dataset[i,1:-1])
plt.scatter([i[0] for i in one_class],[i[1] for i in one_class],c="r",label="Class one")
plt.scatter([i[0] for i in zero_class],[i[1] for i in zero_class],c="b",alpha=0.5,label="Class zero")
plt.legend()
plt.show()


# In[5]:


def evaluate_classification(data,y_predictions):
    y_real = data[:,-1]
    data = data[:,:-1]
    num_points = len(y_real)
    #For preventing zero division errors
    true_positives = 1e-16
    true_negatives = 1e-16
    false_positives = 1e-16
    false_negatives = 1e-16
    for i,real in enumerate(y_real):
        # Trues
        if real == y_predictions[0,i]:
            if y_predictions[0,i] == 1:
                true_positives += 1
            else:
                true_negatives += 1
        else: # Falses
            if y_predictions[0,i] == 1:
                false_positives += 1
            else:
                false_negatives += 1
    
    accuracy  = (true_positives+true_negatives)/num_points
    precision = true_positives / (true_positives+false_positives)
    recall    = true_positives / (true_positives+false_negatives)
    f1        = 2*(recall * precision) / (recall + precision)
    return accuracy, precision, recall, f1


# In[6]:


def exp_normalize(x):
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()

def sigmoid(x):
    z = 1/(1 + exp_normalize(x))
    return z 


# In[ ]:





# In[7]:


def logistic_discriminant(data,calculate_err_each_iter=False,test_data=None):
    train_errors = []
    test_errors = []
    num_points = data.shape[0]
    num_features = len(data[0,:-1])
    w = np.full((1,num_features),random.uniform(-0.01,0.01))
    convergence_point = 1e-18
    old_delta_w = np.full((1,num_features),0)
    count = 1
    while True:
        delta_w = np.full((1,num_features),0)
        o = np.dot(w,data[:,:-1].T)
        y = sigmoid(o)
        if calculate_err_each_iter:
            train_errors.append(tuple(evaluate_classification(data,y)))
            o_test = np.dot(w,test_data[:,:-1].T)
            y_test = sigmoid(o_test)
            test_errors.append(tuple(evaluate_classification(test_data,y_test)))
        # Calculate Delta_w
        delta_w = delta_w + np.dot((np.expand_dims(data[:,-1], axis=0)-y),data[:,:-1])
        w = w - 0.2*delta_w
        dif = np.abs(np.sum(old_delta_w - delta_w)) 
        if np.sum(dif) < convergence_point:
            print("Algorithm reached the convergence after {} iterations".format(count))
            break
        old_delta_w = delta_w.copy() 
        count += 1 
    return w, train_errors, test_errors


# In[ ]:





# In[8]:


# We will perform 10 fold cross validation
one_batch_data = num_points//10
# Divide the dataset to folds
folds = [dataset[i*one_batch_data:(i+1)*one_batch_data] for i in range(10)]
[print("Num of data in the fold {} is {}".format(i+1,len(folds[i]))) for i in range(10)]
accuracy_results = []
precision_results = []
recall_results = []
f1_results = []
print("\n")
for i in range(10):
    print("Fold {} reserved for test".format(i+1))
    # Choose the train and test folds
    test_fold = folds[i]
    train_fold = [folds[j] for j in range(10) if i != j]
    train_fold = np.concatenate(train_fold).ravel().reshape((-1,4))
    # Now train with the 9 train folds and test it on the test fold
    learned_params, train_errors, test_errors = logistic_discriminant(train_fold,True,test_fold)
    class_scores = np.dot(learned_params,test_fold[:,:-1].T)
    y_predictions = sigmoid(class_scores)
    accuracy,precision,recall,f1 = evaluate_classification(test_fold,y_predictions)
    print("Fold {}: Accuracy: {:.4f} ,Precision: {:.4f} ,Recall: {:.4f} ,F1: {:.4f}".format(i+1,accuracy,precision,recall,f1))
    accuracy_results.append(accuracy)
    precision_results.append(precision)
    recall_results.append(recall)
    f1_results.append(f1)
print("\n 10 Fold Cross Validation Results")
print("Accuracy Mean: {:.4f}, STD: {:.4f}".format(np.mean(accuracy_results),np.std(accuracy_results)))
print("Precision Mean: {:.4f}, STD: {:.4f}".format(np.mean(precision_results),np.std(precision_results)))
print("Recall Mean: {:.4f}, STD: {:.4f}".format(np.mean(recall_results),np.std(recall_results)))
print("F1 Mean: {:.4f}, STD: {:.4f}".format(np.mean(f1_results),np.std(f1_results)))

# Latest fold will be used to draw train and test errors at each iterations.
plt.figure()
plt.title("Accuracy vs Iterations")
plt.xlabel("Iterations")
plt.ylabel("Accuracy (%)")
plt.plot([item[0]*100 for item in train_errors] ,label="Train Accuracy")
plt.plot([item[0]*100 for item in test_errors] ,label="Test Accuracy")
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




