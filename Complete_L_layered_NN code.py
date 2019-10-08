#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# ## Initialization of Parameters

# In[2]:


def initialize_parameters(layer_dims): #layer_dims : list of dimensions of each layers in our NN
    np.random.seed(42)
    parameters = {}
    L = len(layer_dims)
    
    for l in range(1,L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    
    return parameters


# In[3]:


params = initialize_parameters([3,4,5])
print('W1 = ' + str(params['W1']))
print('b1 = ' + str(params['b1']))
print('W2 = ' + str(params['W2']))
print('b2 = ' + str(params['b2']))


# ## Forward Propagation

# In[4]:


#helper fuction_1
def linear_forward(A,W,b):
    '''
    A : activations from previous layer --> shape -->(size of prev. layer , number of examples)
    W : weights matrix , numpy array of shape --> (size of current layer , size of prev. layer)
    b : bias vector , numpy array of shape --> (size of current layer , 1)
    
    Returns:
    Z : the input to the activation function
    cache : a python tuple containing 'A', 'W', 'b', stored for efficient back propagation.

    '''
    Z = np.dot(W,A) + b
    cache = (A, W, b)
    
    return Z, cache


# In[5]:


# helper function_2
def sigmoid(z):
    A = 1/(1+np.exp(-z))
    return A,z

def tanh(z):
    A = np.tanh(z)
    return A,z

def relu(z):
    A = np.maximum(0,z)
    return A,z

def leaky_relu(z):
    A = np.maximum(0.01*z , z)
    return A,z


# In[6]:


#helper function_3
def linear_activation_forward(A_prev , W , b, activation):
    '''
    A : activations from previous layer --> shape -->(size of prev. layer , number of examples)
    W : weights matrix , numpy array of shape --> (size of current layer , size of prev. layer)
    b : bias vector , numpy array of shape --> (size of current layer , 1)
    activation: the activation to be used, passed as string.
    
    Returns:
    A : the output of the activation function also called post activation value
    cache : a python tuple conatining 'linear_cache', 'activation_cache'.
    
    '''
    
    if activation == 'sigmoid':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
        
    elif activation == 'relu':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(z)
        
    elif activation == 'tanh':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A , activation_cache = tanh(z)
        
    elif activation == 'leaky_relu':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = leaky_relu(z)


# In[7]:


def L_model_forward(X, parameters):
    '''
    Implementing forward prop. for the [LINEAR->RELU]*(L-1) -> LINEAR->SIGMOID computation,
    can be changed according to need.
    
    X : data, numpy array of shape -->(input_size, number of examples)
    parameters : output of initialize_parameters function.
    
    Returns:
    AL : last post-activation value
    caches : list of caches conataining:
             every cache pf linear_activation_forward()
    '''
    
    caches = []
    A = X
    L = len(parameters) // 2   #gives the number of layers in neural network.
    
    for l in range(1,L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev , parameters['W' + str(l)], parameters['b' + str(l)] , activation = 'relu')
        caches.append(cache)
        
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)] , parameters['b'+ str(L)], activation = 'sigmoid')
    cahes.append(cache)
    
    return AL, caches


# ## Implementing cost function (cross entropy loss)

# In[8]:


def compute_cost(AL, Y):
    '''
    Al = probability vector corresponding to label predictions -->(1 , number of examples)
    Y = true 'label' vector -->(1, number of examples)
    
    '''
    
    m = Y.shape[1]
    cost = -1/m * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1-Y , np.log(1 - AL)))
    
    cost = np.squeeze(cost)      #converts [[15]] to 15 
    return cost


# ## Backward propagation

# In[9]:


def linear_backward(dZ , cache):
    '''
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
   
   '''
    
    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    dW = np.dot(dZ , cache[0].T)/m
    db = (np.sum(dZ, axis = 1, keepdims= True))/m
    dA_prev = np.dot(cache[1], dZ)
    
    return dA_prev, dW, db
    


# In[11]:


def sigmoid_backwards(dA ,cache):
    Z = cache
    dZ = dA*(sigmoid(Z)*(1-sigmoid(Z)))
    return dZ

def relu_backwards(dA,cache):
    Z = cache
    dZ = np.array(dA,copy = True)
    dZ[Z<=0] = 0
    return dZ

def tanh_backwards(dA , cache):
    Z = cache
    dZ = (1-tanh(z))**2
    return dZ


# In[12]:


def linear_activation_backward(dA, cache, activation):
    '''
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    '''
    linear_cache, activation_cache = cache
    if activation == 'relu':
        dZ = relu_backwards(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ,linear_cache)
    
    elif activation == 'sigmoid':
        dZ = sigmoid_backwards(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == 'tanh':
        dZ = tanh_backwards(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    return dA_prev, dW, db
    


# In[13]:


def L_model_backward(AL , Y, caches):
    '''
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    
    '''
    
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    dAL = -(np.divide(Y, AL) - np.divide(1-Y,1 -AL))
    
    current_cache = caches[L-1]
    grads['dA'+str(L-1)] , grads['dW'+str(L)], grads['db'+str(L)] = linear_activation_backward(dAL, current_cache, 'sigmoid')
    
    
    for l in reversed(range(L-1)):
        current_cache = cache[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads['dA'+str(l+1)], current_cache, 'relu')
        grads['dA'+str(l)] = dA_prev_temp
        grads['dW' + str(l+1)] = dW_temp
        grads['db'+str(l+1)] = db_temp
        
    return grads


# In[14]:


def update_parameters(parameters, grads, learning_rate):
    '''
        Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    
    '''
    L = len(parameters)
    
    for l in range(L):
        parameters['W'+str(l+1)] = parameters['W'+str(l+1)] - learning_rate*grads['dW'+str(l+1)]
        parameters['b'+str(l+1)] = parameters['b'+str(l+1)] -learning_rate*grads['db'+str(l+1)]
    return parameters


# # Implementation

# In[15]:


### CONSTANTS ###
layers_dims = [12288, 20, 7, 5, 1] #  4-layer model eg.


# In[17]:


def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations=3000, print_cost=False):
    '''
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.


    '''
    np.random.seed(1)
    costs = []
    
    parameters = initialize_parameters(layers_dims)
    
    for i in range(0,num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL,Y)
        grads = L_model_backward(AL, Y , caches)
        parameters = update_parameters(parameters,grads,learning_rate)
        
    # printing cost after every 100 examples
    if print_cost and i%100 ==0:
        print('Cost after every iteration %i: %f' %(i, cost))
    if print_cost and i%100 == 0:
        costs.append(cost)
    
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
   
    return parameters


# In[19]:


# parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)


# In[ ]:




