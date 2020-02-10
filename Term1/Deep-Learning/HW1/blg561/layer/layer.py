import numpy as np
from abc import ABC, abstractmethod


class Layer(ABC):
    '''
        Abstract layer class which implements forward and backward methods
    '''

    def __init__(self):
        self.x = None

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError('Abstract class!')

    @abstractmethod
    def backward(self, x):
        raise NotImplementedError('Abstract class!')

    def __repr__(self):
        return 'Abstract layer class'


class LayerWithWeights(Layer):
    '''
        Abstract class for layer with weights(CNN, Affine etc...)
    '''

    def __init__(self, input_size, output_size, seed=None):
        if seed is not None:
            np.random.seed(seed)
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


class YourActivation(Layer):
    def __init__(self):
        self.x = None

    def forward(self, x):
        '''
            :param x: outputs of previous layer
            :return: output of activation
        '''
        # Lets have an activation of X^2
        # This activation is similar to sigmoid but negative values 
        # give negative results instead of zero 
        self.x = x.copy()
        out = x * 1/(1+np.exp(-0.8*x))
        return out

    def backward(self, dprev):
        '''
            :param dprev: gradient of next layer:
            :return: downstream gradient
        '''
        # TODO: CHANGE IT
        # Example: derivate of x * sigmoid(-0.8*x)
        dx = (np.exp(0.8*dprev)*(1 + np.exp(0.8*dprev) + 0.8*dprev))/(1 + np.exp(0.8*dprev))**2
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
        probs = None
       
        # Your implementation starts
        # To prevent the zero division make the values always negative
        probs = np.exp(x - np.max(x, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        self.probs = probs.copy()
        # End of your implementation

        return probs

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


class AffineLayer(LayerWithWeights):
    def __init__(self, input_size, output_size, seed=None):
        super(AffineLayer, self).__init__(input_size, output_size, seed=seed)

    def forward(self, x):
        '''
            :param x: activations/inputs from previous layer
            :return: output of affine layer
        '''
        out = None
        # Vectorize the input to [batchsize, others] array
        batch_size = x.shape[0]

        # Do the affine transform
        out = np.dot(x.reshape((batch_size , -1)), self.W)  + self.b

        # Save x for using in backward pass
        self.x = x.copy()

        return out

    def backward(self, dprev):
        '''
            :param dprev: gradient of next layer:
            :return: downstream gradient
        '''

        batch_size = self.x.shape[0]
        # Vectorize the input to a 1D ndarray
        x_vectorized = None
        dx, dw, db = None, None, None

        # YOUR CODE STARTS
        x_vectorized = self.x.reshape(batch_size, -1)
        # dx = dprev* W.T 
        dx = np.dot(dprev, self.W.T).reshape(self.x.shape)
        # dw = w.T * dprev
        dw = np.dot(x_vectorized.T , dprev)
        db = np.sum(dprev,axis=0)
        # YOUR CODE ENDS

        # Save them for backward pass
        self.db = db.copy()
        self.dW = dw.copy()
        return dx, dw, db

    def __repr__(self):
        return 'Affine layer'

class Model(Layer):
    def __init__(self, model=None):
        self.layers = model
        self.y = None

    def __call__(self, moduleList):
        for module in moduleList:
            if not isinstance(module, Layer):
                raise TypeError(
                    'All modules in list should be derived from Layer class!')

        self.layers = moduleList

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, y):
        self.y = y.copy()
        dprev = y.copy()
        dprev = self.layers[-1].backward(y)

        for layer in reversed(self.layers[:-1]):
            if isinstance(layer, LayerWithWeights):
                dprev = layer.backward(dprev)[0]
            else:
                dprev = layer.backward(dprev)
        return dprev

    def __iter__(self):
        return iter(self.layers)

    def __repr__(self):
        return 'Model consisting of {}'.format('/n -- /t'.join(self.layers))

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, i):
        return self.layers[i]

class VanillaSDGOptimizer(object):
    def __init__(self, model, lr=1e-3, regularization_str=1e-4):
        self.reg = regularization_str
        self.model = model
        self.lr = lr

    def optimize(self):
        for m in self.model:            
            if isinstance(m, LayerWithWeights):
                self._optimize(m)

    def _optimize(self, m):
        '''
            Optimizer for SGDMomentum
            Do not forget to add L2 regularization!
            :param m: module with weights to optimize
        '''
        # Your implementation starts
        # Update  learning rate with L2 regularization term
        # In practice regularization only applied to W terms
        m.W -= self.lr * (m.dW + self.reg * m.W)
        m.b -= self.lr * m.db
        # End of your implementation
        return
       
class SGDWithMomentum(VanillaSDGOptimizer):
    def __init__(self, model, lr=1e-3, regularization_str=1e-4, mu=.5):
        self.reg = regularization_str
        self.model = model
        self.lr = lr
        self.mu = mu
        # Save velocities for each model in a dict and use them when needed.
        # Modules can be hashed
        self.velocities = {m: 0 for m in model}

    def _optimize(self, m):
        '''
            Optimizer for SGDMomentum
            Do not forget to add L2 regularization!
            :param m: module with weights to optimize
        '''
        # Your implementation starts
        # Apply regularization
        # In practice regularization only applied to weights
        m.dW += self.reg * m.W

        # Get velocities
        velocities_W = self.velocities[m]
        velocities_b = self.velocities[m]

        # Calculate new velocity     
        velocities_W = self.mu*velocities_W + self.lr*m.dW 
        m.W -= velocities_W 

        m.db += self.reg*m.b
        velocities_b = self.mu*velocities_b + self.lr*m.db
        m.b -= velocities_b

        # End of your implementation