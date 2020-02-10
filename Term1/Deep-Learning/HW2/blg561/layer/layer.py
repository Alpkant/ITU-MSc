#Adapted from Stanford CS231n Course
import numpy as np
from abc import ABC, abstractmethod
from .helpers import flatten_unflatten


class Layer(ABC):
    def __init__(self, input_size, output_size):
        self.W = np.random.rand(input_size, output_size)
        self.b = np.zeros(output_size)
        self.x = None
        self.db = np.zeros_like(self.b)
        self.dW = np.zeros_like(self.W)

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError('Abstract class!')

    @abstractmethod
    def backward(self, x):
        raise NotImplementedError('Abstract class!')

    def __repr__(self):
        return 'Abstract layer class'


class ReLU(Layer):

    def __init__(self):
        # Dont forget to save x or relumask for using in backward pass
        self.x = None

    def forward(self, x):
        '''
            Forward pass for ReLU
            :param x: outputs of previous layer
            :return: ReLU activation
        '''
        # Do not forget to copy the output to object to use it in backward pass
        self.x = x.copy()
        # This is used for avoiding the issues related to mutability of python arrays
        x = x.copy()

        # Implement relu activation
        # x if it's positive else 0
        x = x * (x > 0)
        return x

    def backward(self, dprev):
        '''
            Backward pass of ReLU
            :param dprev: gradient of previos layer:
            :return: upstream gradient
        '''
        dx = None
        # Your implementation starts
        dx = dprev * (self.x > 0)
        # End of your implementation
        return dx


class Softmax(Layer):
    def __init__(self):
        self.probs = None

    def forward(self, x):
        '''
            Softmax function
            :param x: Input for classification (Likelihoods)
            :return: Class Probabilities
        '''
        # Normalize the class scores (i.e output of affine linear layers)
        # In order to avoid numerical unstability.
        # Do not forget to copy the output to object to use it in backward pass
        softmax_scores = None

        # Use your past implementation if needed
        softmax_scores = np.exp(x - np.max(x, axis=1, keepdims=True))
        softmax_scores /= np.sum(softmax_scores, axis=1, keepdims=True)
       
        self.probs = softmax_scores.copy()
        return softmax_scores

    def backward(self, y):
        '''
            Implement the backward pass w.r.t. softmax loss
            -----------------------------------------------
            :param y: class labels. (as an array, [1,0,1, ...]) Not as one-hot encoded
            :return: upstream derivate

        '''
        dx = None
        # Your implementation starts
        dx = self.probs
        # Derivative of softmax fx is: fx * (1-fx)
        dx[np.arange(y.shape[0]), y] -= 1
        dx /= y.shape[0] 
        # End of your implementation

        return dx


def loss(probs, y):
    '''
        Calculate the softmax loss
        --------------------------
        :param probs: softmax probabilities
        :param y: correct labels
        :return: loss
    '''
    loss = None
    
    # Your implementation starts
    num_samples = probs.shape[0]

    # To prevent numerical instability normalize probs by substracting maximum probability
    negative_probs = probs - np.max(probs, axis=1, keepdims=True)
    sum_exponentials = np.sum(np.exp(negative_probs), axis=1, keepdims=True)
    # Take the logs of the sum error
    probs_log = negative_probs - np.log(sum_exponentials)
    
    # Calculate the negative likelihood by using logarithmic probabilities
    loss = -np.sum(probs_log[range(num_samples), y]) / num_samples

    # End of your implementation
    return loss


class Dropout(Layer):
    def __init__(self, p=.5):
        '''
            :param p: dropout factor
        '''
        self.mask = None
        self.mode = 'train'
        self.p = p

    def forward(self, x, seed=None):
        '''
            :param x: input to dropout layer
            :param seed: seed (used for testing purposes)
        '''
        if seed is not None:
            np.random.seed(seed)
        # YOUR CODE STARTS
        if self.mode == 'train':
            out = None

            # Create a dropout mask
            # Implementation is influenced from vanilla dropout http://cs231n.github.io/neural-networks-2/
            
            mask = (np.random.rand( *x.shape )< self.p) / self.p
            
            # Do not forget to save the created mask for dropout in order to use it in backward
            self.mask = mask.copy()

            out = x*mask

            return out
        elif self.mode == 'test':
            out = x*self.mask
            return out
        # YOUR CODE ENDS
        else:
            raise ValueError('Invalid argument!')

    def backward(self, dprev):

        dx = dprev * self.mask
        return dx


class BatchNorm(Layer):
    def __init__(self, D, momentum=.9):
        self.mode = 'train'
        self.normalized = None

        self.x_sub_mean = None
        self.momentum = momentum
        self.D = D
        self.running_mean = np.zeros(D)
        self.running_var = np.zeros(D)
        self.gamma = np.ones(D)
        self.beta = np.zeros(D)
        self.ivar = np.zeros(D)
        self.sqrtvar = np.zeros(D)

    # @flatten_unflatten
    def forward(self, x, gamma=None, beta=None):
        if self.mode == 'train':
            sample_mean = np.mean(x, axis=0)
            sample_var = np.var(x, axis=0)
            if gamma is not None:
                self.gamma = gamma.copy()
            if beta is not None:

                self.beta = beta.copy()

            # Normalise our batch
            self.normalized = ((x - sample_mean) /
                               np.sqrt(sample_var + 1e-5)).copy()
            self.x_sub_mean = x - sample_mean

            # YOUR CODE HERE

            # Update our running mean and variance then store.
            minus_momentum = 1 - self.momentum
            running_mean = self.momentum * self.running_mean + minus_momentum * sample_mean
            running_var = self.momentum * self.running_var + minus_momentum * sample_var
            # Gamma controls the desired stds whereas beta controls desired means
            out = self.gamma * self.normalized + self.beta

            # YOUR CODE ENDS
            self.running_mean = running_mean.copy()
            self.running_var = running_var.copy()

            self.ivar = 1./np.sqrt(sample_var + 1e-5)
            self.sqrtvar = np.sqrt(sample_var + 1e-5)

            return out
        elif self.mode == 'test':
            out = self.gamma * self.normalized + self.beta
        else:
            raise Exception(
                "INVALID MODE! Mode should be either test or train")
        return out

    def backward(self, dprev):
        N, D = dprev.shape
        # YOUR CODE HERE
        dx, dgamma, dbeta = None, None, None
        # Calculate the gradients

        dx = 1.0/N * self.ivar 
        dx = dx * (N*(dprev * self.gamma) - np.sum(dprev * self.gamma, axis=0) - self.normalized*np.sum( (dprev * self.gamma)*self.normalized, axis=0))
        dbeta = np.sum(dprev, axis=0)
        dgamma = np.sum(self.normalized*dprev, axis=0)
           

        return dx, dgamma, dbeta


class MaxPool2d(Layer):
    def __init__(self, pool_height, pool_width, stride):
        self.pool_height = pool_height
        self.pool_width = pool_width
        self.stride = stride
        self.x = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_H = np.int(((H - self.pool_height) / self.stride) + 1)
        out_W = np.int(((W - self.pool_width) / self.stride) + 1)

        self.x = x.copy()

        # Initiliaze the output
        out = np.zeros([N, C, out_H, out_W])

        # Implement MaxPool
        # YOUR CODE HERE
        for n in range(N):
            for out_h in range(out_H):
                for out_w in range(out_W):
                    out[n, :, out_h, out_w] = np.max(x[n, :, out_h*self.stride:out_h*self.stride+self.pool_height,out_w*self.stride:out_w*self.stride+self.pool_width], axis=(1, 2))

        return out

    def backward(self, dprev):
        x = self.x
        N, C, H, W = x.shape
        _, _, dprev_H, dprev_W = dprev.shape

        dx = np.zeros_like(self.x)

        # Calculate the gradient (dx)
        # YOUR CODE HERE
        # I couldn't implemented
        return dx


class Flatten(Layer):
    def __init__(self):
        self.N, self.C, self.H, self.W = 0, 0, 0, 0

    def forward(self, x):
        self.N, self.C, self.H, self.W = x.shape
        out = x.reshape(self.N, -1)
        return out

    def backward(self, dprev):
        return dprev.reshape(self.N, self.C, self.H, self.W)
