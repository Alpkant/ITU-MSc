# BLG527E Machine Learning Homework2
# Author: Alperen Kantarcı
# No: 504191504

import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
from matplotlib.colors import ListedColormap


# Get the data points
dataset = loadmat('hw2.mat')['d']

num_points = dataset.shape[0]
num_class1 = np.sum(dataset,axis=0)[-1]
num_class0 = num_points - num_class1
print("P(c=0) = {}, P(c=1) = {}".format(num_class0/num_points,num_class1/num_points))




# Shuffle data to choose sets randomly
# Note shuffle only shuffles rows(data points), it doesn't change order of the features
np.random.shuffle(dataset)
train_set = dataset[:int(0.8*num_points)]
test_set = dataset[int(0.8*num_points):]
print("Train set size: {} , Test set size: {}".format(train_set.shape,test_set.shape))




# Take only features into account
train_only_features = train_set[:,:-1]
train_labels = train_set[:,-1:]
def mean(mat):
    mean_vec = np.zeros(mat.shape[1])
    for ind,i in enumerate(mat):
        mean_vec += i
    
    mean_vec = mean_vec/mat.shape[0]
    return mean_vec

def covariance(mat,mean):
    num_feature = mat.shape[1]
    cov_matrix = np.zeros((num_feature,num_feature))
    for i in range(num_feature):
        for j in range(num_feature):
            cov_matrix[i][j] = np.sum( (mat[:,i]-mean[i])*(mat[:,j]-mean[j]) )/(mat.shape[0]-1) 
    return cov_matrix
   
# Calculate shared mean and shared covariance matrix for data with all classes
# Calculate priors for training set
num_points = train_set.shape[0]
num_class1 = np.sum(train_set,axis=0)[-1]
num_class0 = num_points - num_class1

priors = np.array([[num_class0/num_class1],[num_class1/num_class0]])
shared_mean_val = mean(train_only_features)
shared_covariance_mat = covariance(train_only_features,shared_mean_val)
print("Shared Mean of x1 (feature1): {:.7}, x2 (feature2): {:.7}".format(shared_mean_val[0],shared_mean_val[1]))
print("Shared Covarince matrix:\n", shared_covariance_mat)

# Partition data to each class to compute mean and covariance matrixes
class_0_x = train_only_features[np.where(train_labels == 0)[0]] 
class_0_y = train_labels[np.where(train_labels == 0)[0]]
class_1_x = train_only_features[np.where(train_labels == 1)[0]]
class_1_y = train_labels[np.where(train_labels == 1)[0]]
#Calculate mean and covariance matrixes for each class
class_0_mean = mean(class_0_x) 
class_0_cov = covariance(class_0_x,class_0_mean)
class_1_mean = mean(class_1_x)
class_1_cov = covariance(class_1_x,class_1_mean)
mu_matrix = np.vstack((class_0_mean,class_1_mean))
print("\nClass 0 Mean of x1 (feature1): {:.7}, x2 (feature2): {:.7}".format(class_0_mean[0],class_0_mean[1]))
print("Class 0 Covarince matrix:\n", class_0_cov)
print("\nClass 1 Mean of x1 (feature1): {:.7}, x2 (feature2): {:.7}".format(class_1_mean[0],class_1_mean[1]))
print("Class 1 Covarince matrix:\n", class_1_cov)



def lda( x, mu, sigma, prior):
    num_class = mu.shape[0]
    inv_sigma = np.linalg.inv(sigma)
    results = np.zeros((x.shape[0],num_class))
    # For every class label calculate classification score
    for j in range(num_class):
        term1 = np.dot(np.dot(x,inv_sigma),mu[j].T)
        term2 = -.5*mu[j].T@inv_sigma@mu[j]
        term3 = np.log(prior[j])
        results[:,j] = term1+term2+term3
    return  results
 
def qda( x, mu, sigmas, prior ):
    num_class = mu.shape[0]
    inv_sigma_list = [np.linalg.inv(sig) for sig in sigmas]
    results = np.zeros((x.shape[0],num_class))
    # Qda applied without vectorization,to show that mods can be done without vectorizing
    for i in range(x.shape[0]):
    # For every class label calculate classification score
        for j in range(num_class):
            term1 = np.float32(-.5*np.log(np.linalg.det(sigmas[j])))
            term2 = -.5*(x[i]-mu[j]).T@inv_sigma_list[j]@(x[i]-mu[j])
            term3 = np.log(prior[j])
            results[i,j] = term1+term2+term3
    return  results

# To calculate the boundary two class probability should be equal. Solving the equation will give boundary
def calculate_boundary_linear(x,mu,sigma,prior):
    x = x.T
    return (np.log(prior[0] / prior[1]) - 1/2 * (mu[0] + mu[1]).T @ np.linalg.inv(sigma)@(mu[0] - mu[1]) + x.T @ np.linalg.inv(sigma)@ (mu[0] - mu[1]))

def calculate_boundary_quadratic(x,mu,sigma,prior):
    return (np.log(prior[0] / prior[1]) -1/2 * np.log(np.linalg.det(sigma[0])/np.linalg.det(sigma[1])) - 1/2 * (x-mu[0])@np.linalg.inv(sigma[0])@(x-mu[0]).T 
            + 1/2 * (x-mu[1])@np.linalg.inv(sigma[1])@(x-mu[1]).T)


#### Question 2 QDA ####
# Train classification
train_prediction_scores = qda(train_only_features,mu_matrix,[class_0_cov,class_1_cov],priors)
train_prediction = np.argmax(train_prediction_scores,axis=1)

# Calculate training error 
true = np.sum(train_labels[:,0]==train_prediction)
total = train_labels.shape[0]
train_acc = "QDA Training accuracy : {}% True:{} False:{}".format(100*true/total,true,total-true)
print(train_acc)

# Test classification
test_features = test_set[:,:-1]
test_labels = test_set[:,-1:]
test_prediction_scores = qda(test_features,mu_matrix,[class_0_cov,class_1_cov],priors)
test_prediction = np.argmax(test_prediction_scores,axis=1)
# Calculate test error
true = np.sum(test_labels[:,0]==test_prediction)
total = test_labels.shape[0]
test_acc = "QDA Test accuracy : {}% True:{} False:{}".format(100*true/total,true,total-true)
print(test_acc)

# Calculate decision boundary
x = np.arange(-6.0, 4, 5e-1)
y = np.arange(-6.0, 8, 5e-1)
X, Y = np.meshgrid(x, y)
quadratic_boundary = np.array([ calculate_boundary_quadratic(np.array([xx,yy]).reshape(-1,2),mu_matrix,[class_0_cov,class_1_cov],priors)
                     for xx, yy in zip(np.ravel(X), np.ravel(Y)) ]).reshape(X.shape)



colors = ['blue','red']
## Draw train set ##
# Draw train set with real labels and decision boundary
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(30,10))
fig.text(0.5, .95, train_acc, ha='center', va='top')
ax1.set_title("Real labels of train set by Quadratic Discriminant Classifer")
ax1.contour( X, Y, quadratic_boundary , levels = [0],cmap=ListedColormap("black") )
ax1.scatter(train_only_features[:,0], train_only_features[:,1],s=16,c=train_labels[:,0], cmap=ListedColormap(colors))

# Draw train set only predictions
ax2.set_title("Assigned labels of train set by according to the predictions of Quadratic Discriminant Classifer")
ax2.contour( X, Y, quadratic_boundary , levels = [0],cmap=ListedColormap("black") )
ax2.scatter(train_only_features[:,0], train_only_features[:,1],s=16,c=train_prediction, cmap=ListedColormap(colors))
plt.show()


# Draw decision boundary

## Draw test set ##
#Draw test set with real labels and decision boundary
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(30,10))
fig.text(0.5, .95, test_acc, ha='center', va='top')
ax1.set_title("Real labels of test set by Quadratic Discriminant Classifer")
ax1.contour( X, Y, quadratic_boundary , levels = [0],cmap=ListedColormap("black") )
ax1.scatter(test_features[:,0], test_features[:,1],s=16,c=test_labels[:,0], cmap=ListedColormap(colors))

#Draw test set only predictions

ax2.set_title("Assigned labels of test set by Quadratic Discriminant Classifer")
ax2.contour( X, Y, quadratic_boundary , levels = [0],cmap=ListedColormap("black") )
ax2.scatter(test_features[:,0], test_features[:,1],s=16,c=test_prediction, cmap=ListedColormap(colors))
plt.show()


#### Question 3 LDA #####
# Train classification
train_prediction_scores = lda(train_only_features,mu_matrix,shared_covariance_mat,priors)
train_prediction = np.argmax(train_prediction_scores,axis=1)
# Calculate training error 
true = np.sum(train_labels[:,0]==train_prediction)
total = train_labels.shape[0]
train_acc = "LDA Training accuracy : {}% True:{} False:{}".format(100*true/total,true,total-true)
print(train_acc)

# Test classification
test_features = test_set[:,:-1]
test_labels = test_set[:,-1:]
test_prediction_scores = lda(test_features,mu_matrix,shared_covariance_mat,priors)
test_prediction = np.argmax(test_prediction_scores,axis=1)
# Calculate test error
true = np.sum(test_labels[:,0]==test_prediction)
total = test_labels.shape[0]
test_acc = "LDA Test accuracy : {}% True:{} False:{}".format(100*true/total,true,total-true)
print(test_acc)

# Calculate boundaries for both train and test
x = np.arange(-6.0, 4, 5e-1)
y = np.arange(-6.0, 8, 5e-1)
X, Y = np.meshgrid(x, y)
boundary = np.array([ calculate_boundary_linear(np.array([xx,yy]).reshape(-1,2),mu_matrix,shared_covariance_mat,priors)
                     for xx, yy in zip(np.ravel(X), np.ravel(Y)) ]).reshape(X.shape)



colors = ['blue','red']
## Draw train set ##
# Draw train set with real labels and decision boundary
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(30,10))
fig.text(0.5, .95, train_acc, ha='center', va='top')
ax1.set_title("Real labels of train set by Linear Discriminant Classifer")
ax1.contour( X, Y, boundary , levels = [0],cmap=ListedColormap("black") )
ax1.scatter(train_only_features[:,0], train_only_features[:,1],s=16,c=train_labels[:,0], cmap=ListedColormap(colors))

# Draw train set only predictions
ax2.set_title("Assigned labels of train set by Linear Discriminant Classifer")
ax2.contour( X, Y, boundary , levels = [0],cmap=ListedColormap("black") )
ax2.scatter(train_only_features[:,0], train_only_features[:,1],s=16,c=train_prediction, cmap=ListedColormap(colors))
plt.show()
# Draw decision boundary

## Draw test set ##
#Draw test set with real labels and decision boundary
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(30,10))
ax1.set_title("Real labels of test set by Linear Discriminant Classifer")
fig.text(0.5, .95, test_acc, ha='center', va='top')
ax1.contour( X, Y, boundary , levels = [0],cmap=ListedColormap("black") )
ax1.scatter(test_features[:,0], test_features[:,1],s=16,c=test_labels[:,0], cmap=ListedColormap(colors))

#Draw test set only predictions
ax2.set_title("Assigned labels of test set by Linear Discriminant Classifer")
ax2.contour( X, Y, boundary , levels = [0],cmap=ListedColormap("black") )
ax2.scatter(test_features[:,0], test_features[:,1],s=16,c=test_prediction, cmap=ListedColormap(colors))
plt.show()



#Question 5
dim = 2
example = 1000
mu1  = np.array([1, 2]).reshape(2, 1)
mu2  = np.array([3, 5]).reshape(2, 1)
cov1 = np.array([[2, 1],
                 [1, 3]])
cov2 = np.array([[1, -0.8],
                 [-0.8, 3]])

# Compute Cholesky decomposition to get upper triangular matrix
def cholesky(A):
    L = [[0.0] * len(A) for _ in range(len(A))]
    for i, (Ai, Li) in enumerate(zip(A, L)):
        for j, Lj in enumerate(L[:i+1]):
            s = sum(Li[k] * Lj[k] for k in range(j))
            Li[j] = np.sqrt(Ai[i] - s) if (i == j) else (1.0 / Lj[j] * (Ai[j] - s))
    return np.array(L)

# Step 1, 1-dimensional normal distribution with mean 0 and variance 
Y = np.random.normal(loc=0, scale=1, size=dim*example).reshape(dim, example)

# Step 2, determine the upper triangular Cholesky factor 

R1 = cholesky(cov1)
print("R1:\n",R1.T)
result = np.dot(R1.T, R1)
# result should be close to the original covariance
# if cholesky decomposition calculation is correct
print("R1.T*R1 :\n",result)
print("Cov:",cov1)

R2 = cholesky(cov2)
result = np.dot(R2.T, R2)
print("R2.T*R2 :\n",result)
print("Cov:\n",cov2)

# Step 3, compute X = MU + R.T * Y
x1 = (mu1 + np.dot(R1, Y)).T
x2 = (mu2 + np.dot(R2, Y)).T
fig = plt.figure(figsize=(16,10))
plt.scatter(x1[:,0],x1[:,1],c='blue')
plt.scatter(x2[:,0],x2[:,1],c='red')
plt.title("Generated Dataset with given means and covariances.")
plt.show()