import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import time


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return  1.0/(1.0+np.exp(-1.0*z))


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary
    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples. 
    # Your code here.

    train_data = np.array([])
    train_label = np.array([])
    validation_data = np.array([])
    validation_label = np.array([])
    test_data = np.array([])
    test_label = np.array([])

    # getting test data
    for i in range(10):
        test_i = mat['test' + str(i)]
        if len(test_data) == 0:
            test_data = test_i
        else:
            test_data = np.vstack((test_data, test_i))
        test_label = np.hstack((test_label, np.full(len(test_i), i, dtype=int)))

    # getting train data
    for i in range(10):
        train_i = mat['train' + str(i)]
        if len(train_data) == 0:
            train_data = train_i
        else:
            train_data = np.vstack((train_data, train_i))
        train_label = np.hstack((train_label, np.full(len(train_i), i, dtype=int)))

    # Feature selection
    # Your code here.

    # dimReduce contains index of reduced features
    # features reduced from 784 to 717 for train data
    dimReduce = []
    train_data = train_data.T
    for i in range(784):
        a = np.unique(train_data[i])
        if len(a) <= 1:
            dimReduce.append(i)
    dimReduce.reverse()
    for i in range(len(dimReduce)):
        train_data = np.delete(train_data, dimReduce[i], 0)
    train_data = train_data.T

    testReduce=[]
    test_data=test_data.T
    for i in range(784):
        a=np.unique(test_data[i])
        if len(a) <= 1:
            testReduce.append(i)
    testReduce.reverse()
    for i in range(len(testReduce)):
        test_data=np.delete(test_data, testReduce[i], 0)
    test_data=test_data.T

    # getting 10000 validation data from train data
    # change range(10) to range(10000) for real testing
    for i in range(10000):
        index = np.random.randint(0, len(train_label))
        if len(validation_data) == 0:
            validation_data = train_data[index, :]
        else:
            validation_data = np.vstack((validation_data, train_data[index, :]))
        validation_label = np.hstack((validation_label, train_label[index]))
        train_data = np.delete(train_data, index, 0)
        train_label = np.delete(train_label, index)

    # print (train_data.shape)
    # print("preprocess end")

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    #print ("Step 1>>>>>")
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1))) # 50x718
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1))) # 10x51
    obj_val = 0
    # Your code here

    #So to think about it w1 is from input to hidden layer and w2 is hidden layer to output
    #So we need to calculate the weight sum if I am not mistaken
    #Should be weight1 * input + weight2* input,,,

    # add bias to the end
    temp = np.full((1,len(training_label)), 1, dtype=int) 
    temp = temp.reshape(-1,1) # 5x1
    #training_data = training_data.reshape(-1,1)
    training_data = np.column_stack((training_data,temp)) # 5x718
    # initialize hidden layer output
    o = np.array([]) # 5x10
    loss = 0
    
    grad_hidden_layer = np.zeros((10, n_hidden+1), int)
    grad_input_layer = np.zeros((n_hidden, n_input+1), int)

    for i in range(len(training_label)):
        output_i = []
        input_i = training_data[i, :]
        # find output of all hidden layer and store in z
        z = []
        for j in range(n_hidden):
            w1j = w1[j, :]
            w1j.reshape(-1, 1)
            output = np.dot(input_i.T, w1j)
            z.append(sigmoid(output))
        z.append(1)
        # find output of all output layer and store in o
        for j in range(n_class):
            w2j = w2[j, :]
            w2j.reshape(-1, 1)
            output = np.dot(z, w2j)
            output_i.append(sigmoid(output))
        if len(o) == 0:
            o = output_i
        else:
            o = np.vstack((o, output_i))
    
        # Calculate the loss:-
        true_label = training_label[i]
        y_l = [0] * 10
        y_l[int(true_label)] = 1 # For 1-of-k coding scheme
        temp_sum = 0
        for k in range(len(output_i)):
            o_l = output_i[k] # Output at kth node
            temp_sum += (y_l[k]*np.log(o_l)) + ((1-y_l[k])*np.log(1-o_l))

        # # Vectorized way
        # temp_sum += (y_l * np.log(output_i))

        # Add the error for this training instance to the total loss
        loss += temp_sum

        # Calculate the gradients
        # Gradients for hidden layer to output layer:-
        grad_hidden_temp = np.empty((10,0), int)
        total_hidden = n_hidden+1
        output_error = np.subtract(output_i, y_l)
        for node in range(total_hidden):
            grad_hidden_temp = np.append(grad_hidden_temp, np.array([output_error * z[node]]).transpose(), axis=1)

        # Add the gradient matrix calculated for this training instance to the main gradient matrix for hidden layers.
        grad_hidden_layer = np.add(grad_hidden_layer, grad_hidden_temp)


        #Gradients for input layer to hidden layer:-
        grad_input_temp = np.empty((n_hidden,0), int)
        total_inputs = n_input+1
        z_new = np.array(z[:-1])
        
        output_error = np.array(output_error).transpose()
        for in_node in range(total_inputs):
            grad_jp = []
            for hidden_node in range(n_hidden): # All hidden nodes except bias
                val = np.multiply((1-z_new[hidden_node]), z_new[hidden_node]) * (np.dot(w2[:, hidden_node], output_error)) * input_i[in_node]
                grad_jp.append(val)
            grad_input_temp = np.append(grad_input_temp, np.array([grad_jp]).transpose(), axis=1)
        grad_input_layer = np.add(grad_input_layer, grad_input_temp)


    loss = (-1*loss)/len(training_label)

    # Regularization for loss
    loss += (lambdaval*(np.sum(w1*w1) + np.sum(w2*w2)))/(2*len(training_label))

    # Regularization for gradients:-
    grad_input_layer = (grad_input_layer + (w1*lambdaval))/len(training_label)
    grad_hidden_layer = (grad_hidden_layer + (w2*lambdaval))/len(training_label)

    #print ("loss is", loss)
    # print ("grad for hidden layer", grad_hidden_layer)
    # print ("grad for input layer", grad_input_layer)

    #print(len(z))
    #print("dimension checks")

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    obj_grad = np.concatenate((grad_input_layer.flatten(), grad_hidden_layer.flatten()),0)
    # print ("type obj_grad", type(obj_grad))
    # print ("obj_grad shape", obj_grad.shape)
    #obj_grad = np.array([])
    return (loss, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""
    labels = np.array([])
    # Your code here
    temp = np.full((1,data.shape[0]), 1, dtype=int) 
    temp = temp.reshape(-1,1)
    #data = data.reshape(-1,1)
    data = np.column_stack((data,temp))
    #predicting first thousand labels, change to range(data.shape[0]) in final submission
    for i in range(data.shape[0]):
        output_i = []
        input_i = data[i, :]
        # find output of all hidden layer and store in z
        z = []
        for j in range(n_hidden):
            w1j = w1[j, :]
            w1j.reshape(-1, 1)
            output = np.dot(input_i.T, w1j)
            z.append(sigmoid(output))
        z.append(1)
        # find output of all output layer and store in o
        for j in range(n_class):
            w2j = w2[j, :]
            w2j.reshape(-1, 1)
            output = np.dot(z, w2j)
            output_i.append(sigmoid(output))
        maxval = 0
        idx=0
        for k in range(len(output_i)):
            if output_i[k] > maxval:
                maxval=output_i[k]
                idx = k
        if len(labels) <= 0:
            labels=np.array([idx])
        else:
            labels = np.hstack((labels, idx))
    #print("label predicting end")
    return labels


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

print ("train_data", train_data.shape)
print ("test_data", test_data.shape)
print ("validation_data", validation_data.shape)
# #  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

## set the number of nodes in hidden unit (not including bias unit)
#n_hidden = 50

# # set the number of nodes in output unit
n_class = 10

# # initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)


# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter': 50}  # Preferred value.

# Taking only 5 rows. This is just to test the code. 
train_data_small = train_data[2000:2005, :]
train_label_small = train_label[2000:2005]

# Hidden layer units and lambda
hidden_units = [4,8,12,16,20,50]
lambdaval_list = [0,10,20,30,40,50,60] 
validation_error_params = []
best_params = None

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)

for n_hidden in hidden_units:
    for lambdaval in lambdaval_list:
        args = (n_input, n_hidden, n_class, train_data_small, train_label_small, lambdaval)

        start = time.time()
        nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
        end = time.time()
        time_to_train = end-start
        print ("time taken to train", time_to_train)

        # Reshape nnParams from 1D vector into w1 and w2 matrices
        w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
        w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

        #Test the computed parameters

        predicted_label = nnPredict(w1, w2, train_data)
        #find the accuracy on Training Dataset
        train_accuracy = np.mean((predicted_label == train_label).astype(float))
        print('\n Training set Accuracy:' + str(100 * train_accuracy) + '%')

        predicted_label = nnPredict(w1, w2, validation_data)
        # find the accuracy on Validation Dataset
        predicted_accuracy = np.mean((predicted_label == validation_label).astype(float))
        print('\n Validation set Accuracy:' + str(100 * predicted_accuracy) + '%')
        
        # This will store results of each combination, so we can plot the results later.
        validation_error_params.append([n_hidden, lambdaval, train_accuracy, predicted_accuracy])

        if best_params is None:
            best_params = [w1, w2, n_hidden, lambdaval, train_accuracy, predicted_accuracy]
        else:
            if best_params[5] <= predicted_accuracy:
                best_params = [w1, w2, n_hidden, lambdaval, train_accuracy, predicted_accuracy]

# Get the test error on the weights found from the best hyperparameters
w1, w2 = best_params[0], best_params[1] 
predicted_label = nnPredict(w1, w2, test_data)
# find the accuracy on Validation Dataset
print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

