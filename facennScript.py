'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''
from scipy.optimize import minimize
from math import sqrt
import numpy as np
import pickle
import time

# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon
    return W



# Replace this with your sigmoid implementation
def sigmoid(z):
    return  1.0/(1.0+np.exp(-1.0*z))
    

# Replace this with your nnObjFunction implementation
def nnObjFunction(params, *args):
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1))) 
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    # Your code here

    #So to think about it w1 is from input to hidden layer and w2 is hidden layer to output
    #So we need to calculate the weight sum if I am not mistaken
    #Should be weight1 * input + weight2* input,,,

    # add bias to the end
    temp = np.full((1,len(training_label)), 1, dtype=int) 
    temp = temp.reshape(-1,1)
    #training_data = training_data.reshape(-1,1)
    training_data = np.column_stack((training_data,temp))
    # initialize hidden layer output
    loss = 0
    
    grad_hidden_layer = np.zeros((n_class, n_hidden+1), int)
    grad_input_layer = np.zeros((n_hidden, n_input+1), int)

    for i in range(len(training_label)):
        input_i = training_data[i, :]
        # find output of all hidden layer and store in z
        output = np.dot(w1, input_i.T)
        z = sigmoid(output)
        z = np.append(z, 1)

        # find output of all output layer and store in o
        output = np.dot(w2, z.T)
        out_np = sigmoid(output)
    
        # Calculate the loss:-
        true_label = training_label[i]
        y_l = [0] * n_class
        y_l[int(true_label)] = 1 # For 1-of-k coding scheme

        # Vectorized way
        yl_np = np.array(y_l)
        loss += np.sum(np.multiply(yl_np, np.log(out_np)) + np.multiply(1-yl_np, np.log(1-out_np)))


        # Calculate the gradients

        # Gradients for hidden layer to output layer:-
        # Vectorized way
        z = z.tolist()
        delta = out_np - yl_np
        grad_hidden_temp = (delta.reshape(-1,1) * z)

        # Add the gradient matrix calculated for this training instance to the main gradient matrix for hidden layers.
        grad_hidden_layer = np.add(grad_hidden_layer, grad_hidden_temp)
 
        # Gradients for input layer to hidden layer:-
        z_new = np.array(z[:-1])
        z_new = z_new.reshape(-1,1)
        w2_new = np.delete(w2, n_hidden-1, 1)
        grad_input_temp = (((1-z_new)*z_new) * (np.dot(delta, w2_new)).reshape(-1,1)) * input_i
        grad_input_layer = np.add(grad_input_layer, grad_input_temp)

    loss = (-1*loss)/len(training_label)

    # Regularization for loss
    loss += (lambdaval*(np.sum(w1*w1) + np.sum(w2*w2)))/(2*len(training_label))

    # Regularization for gradients:-
    grad_input_layer = (grad_input_layer + (w1*lambdaval))/len(training_label)
    grad_hidden_layer = (grad_hidden_layer + (w2*lambdaval))/len(training_label)

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    obj_grad = np.concatenate((grad_input_layer.flatten(), grad_hidden_layer.flatten()),0)
    return (loss, obj_grad)

    
# Replace this with your nnPredict implementation
def nnPredict(w1,w2,data):
    labels = np.array([])
    # Your code here
    temp = np.full((1,data.shape[0]), 1, dtype=int) 
    temp = temp.reshape(-1,1)
    #data = data.reshape(-1,1)
    data = np.column_stack((data,temp))
    #predicting first thousand labels, change to range(data.shape[0]) in final submission
    rows = data.shape[0]
    for i in range(rows):
        #output_i = []
        input_i = data[i, :]
        # find output of all hidden layer and store in z
        output = np.dot(w1, input_i.T)
        z = sigmoid(output)
        z = np.append(z, 1)

        output = np.dot(w2, z.T)
        output_i = sigmoid(output)
        output_i = output_i.tolist()

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
    return labels



# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

"""**************Neural Network Script Starts here********************************"""
para = open('params', 'wb')

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
print ("n_input", n_input)

# set the number of nodes in output unit
n_class = 2


# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter': 50}  # Preferred value.

# Hidden layer units and lambda
hidden_units = [4,8,12,16,20]
lambdaval_list = [0,10,20,30,40,50,60] 
# hidden_units = [5, 6]
# lambdaval_list = [0, 5] 
validation_error_params = []
best_params = None

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)

for n_hidden in hidden_units:
    # initialize the weights into some random matrices
    initial_w1 = initializeWeights(n_input, n_hidden)
    initial_w2 = initializeWeights(n_hidden, n_class)

    # unroll 2 weight matrices into single column vector
    initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)
    for lambdaval in lambdaval_list:
        args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

        start = time.time()
        nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
        end = time.time()
        time_to_train = end-start
        ##print ("time taken to train", time_to_train)

        # Reshape nnParams from 1D vector into w1 and w2 matrices
        w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
        w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

        #Test the computed parameters

        predicted_label = nnPredict(w1, w2, train_data)
        #find the accuracy on Training Dataset
        train_accuracy = np.mean((predicted_label == train_label).astype(float))
        #print('\n Training set Accuracy:' + str(100 * train_accuracy) + '%')

        predicted_label = nnPredict(w1, w2, validation_data)
        # find the accuracy on Validation Dataset
        predicted_accuracy = np.mean((predicted_label == validation_label).astype(float))
        #print('\n Validation set Accuracy:' + str(100 * predicted_accuracy) + '%')
        
        # This will store results of each combination, so we can plot the results later.
        validation_error_params.append([w1, w2, n_hidden, lambdaval, train_accuracy, predicted_accuracy, time_to_train])

        if best_params is None:
            best_params = [w1, w2, n_hidden, lambdaval, train_accuracy, predicted_accuracy]
        else:
            if best_params[5] <= predicted_accuracy:
                best_params = [w1, w2, n_hidden, lambdaval, train_accuracy, predicted_accuracy]


# Get the test error on the weights found from the best hyperparameters
w1, w2 = best_params[0], best_params[1] 
predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Test Dataset
print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
# 86.94% with n_hidden = 4, lambdaval = 50

# Here is the part where dump file into pickle file
pickle.dump(best_params[2], para)
pickle.dump(w1, para)
pickle.dump(w2, para)
pickle.dump(best_params[3], para)   # Best combination {4 hidden units, lambdaval = 50}
para.close()


#print ("\n details of params", validation_error_params)

# Store the results of all the hyperparameter combination
hyper_params = open('all_hyperparams', 'wb')
pickle.dump(validation_error_params, hyper_params)
hyper_params.close()


