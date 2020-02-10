# BLG527E Machine Learning Homework1
# Author: Alperen Kantarci
# No: 504191504
# Question 2

# To run this script execute following command.
python Question2.py

# Required libraries are imported
import numpy as np
import warnings
from matplotlib import pyplot as plt
# Pyplot gives error for axes editing
# For better visualization supressing is required
warnings.filterwarnings("ignore")
# Get the data points
x = [0.2, 0.5, 0.4, 0.7, 0.8, 0.9, 1]
y = [0.0, 0.4, 0.5, 0.6, 0.7, 0.9, 1.1]
num_points = len(x)

####### Question 2a #######
# This method only calculates line for each polynomial degree
# For this question degree should be 1 as we use linear model

# Coefficient estimation via matrix 
# y = (X^t X)^-1 + b
# small x holds x values of data
# The i-th row of X will contain the x value for the i-th data sample

def polynomial_regression(x,y,degree = 1):
    if degree >= len(x):
        raise Exception("Degree cannot be higher than (number of data points - 1)")

    # Create column vector of y
    y = np.array(y).reshape(len(y),1)
    # Create big x matrix which holds values
    X = np.ones((len(x),degree+1))
    for row_ind,row in enumerate(X):
        for column_i in range(len(row)):
            row[column_i] = x[row_ind]**column_i
    b_vect =  np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)
    return b_vect
# Make predictions by given points
predict_points = lambda x,b_vect: np.sum([beta*(x**i) for i,beta in enumerate(b_vect)])

# Calculate a and b for the data
b_vect = polynomial_regression(x,y,1) # Linear model
# Plot data points
plt.figure("Linear regression with degree 1")
plt.scatter(x, y, c='blue', label='Data Points', marker = "o")
# Plot line
line_points = np.linspace(0.0, 1.02, 1000)
line = list(map(predict_points,line_points,len(line_points)*[b_vect]))
plt.plot(line_points, line, color='red', label='Regression Line')
axes = plt.axes()
axes.set_ylim([-0.2, 1.2])
axes.set_xlim([0, 1.02])
plt.xlabel('x(t)')
plt.ylabel('y(t)')
plt.xlabel('x(t)')
plt.ylabel('y(t)')
plt.legend()
plt.show()

####### Question 2b #######
# We have calculated b values therefore only writing x value is enough

print("Prediction for x=-3 is: {:.3f}".format(predict_points(-3,b_vect)))
print("Prediction for x=5 is: {:.3f}".format(predict_points(5,b_vect)))

####### Question 2c #######
b_vect  = polynomial_regression(x,y,4)

# Plot data points
plt.figure("Polynomial regression with degree 4")
plt.scatter(x, y, c='blue', label='Data Points', marker = "o")
# Plot line
line_points = np.linspace(0.0, 1.2, 1000)
# Estimate the y value for every point in the x axes
line = list(map(predict_points,line_points,len(line_points)*[b_vect]))
plt.plot(line_points, line, color='red', label='Regression Line')
# Set scaling to the same for every polynomial degree
axes = plt.axes()
axes.set_ylim([-0.2, 1.2])
axes.set_xlim([0, 1.02])
plt.xlabel('x(t)')
plt.ylabel('y(t)')
plt.legend()
plt.show()

# In order to assess the variances of each model
# For each model variance of the errors with leave one out method
# calculated and returned as scalar value
    
def leave_one_out(x,y,degree):
    errors = []
    for ind,x_i in enumerate(x):
        true_y = y[ind]
        remaining_x = np.delete(x,ind)
        remaining_y = np.delete(y,ind)
        # Train the model with data except the leaved
        b_vect = polynomial_regression(remaining_x,remaining_y,degree)
        errors.append( (true_y - predict_points(x_i,b_vect))**2 )
    
    return np.var(errors)


print("Leave one out variance for polynomial degree 1 (linear) :{}".format(leave_one_out(x,y,1)))
print("Leave one out variance for polynomial degree 4 :{}".format(leave_one_out(x,y,4)))
