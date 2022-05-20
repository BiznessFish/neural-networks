import numpy as np
import pandas as pd
from toolkit import evaluate_reg

class MLPReg:
    
    

    def __init__(self, layers, epochs, learning_rate = .2 , show_training_plot = False):
        """
        A multilayer feed forward network that outputs regression.
        
        Parameters:
        -------------
        layers : array of ints. The number of values will determine the number of nodes at each layer. The output layer will have one value. The inuput layer must match up with the dimension of the dataset 
        epochs : int of epochs used in training
        learning_rate : float of eta, or step size, for traiing. Defaults to 0.2 
        show_training_plot : bool of whether or not to plot training/validation curves. Defaults to False        
        
        
        Attributes:
        --------------
        weights : nd-array of floats. Weights correspond to each node of each layer of the neural network. Does not exist if .fit() has not been caleld.
        biases : array of floats. Biases corresponding to each layer of of the neural network. Does not exist if .fit() has not been called.
        
        
        """
        
        self.learning_rate = learning_rate
        self.layers = layers
        self.epochs = epochs
        self.show_training_plot = show_training_plot
        
        # here are the attributes we'll learn from the data
        self.weights = []
        self.biases = []
        
        for i in range(len(self.layers)):
            if i == len(self.layers)-1:
                
                # since the final layer maps to one value, we only need this one set of weights and one single bias

                self.weights.append(np.random.default_rng().uniform(-0.01, 0.01, (1,self.layers[i])))
                self.biases.append(np.random.default_rng().uniform(-0.01, 0.01, (1,)))
            else:

                self.weights.append(np.random.default_rng().uniform(-0.01, 0.01, (self.layers[i+1], self.layers[i])))
                self.biases.append(np.random.default_rng().uniform(-0.01, 0.01, self.layers[i + 1]))

    
    def fit(self, X, y, validation_X, validation_y):
        """
        Function to train to feed forward network. Implements the backpropagation rule. 
        
        Paramters:
        
        X : pandas Dataframe of the feature variables.
        y : pandas Series of the associated outputs for each row of the data
        validation_X : pandas DataFrame of validation data features
        validation_y : pandas Series of validation real outputs
        
        """
        
        
        
        
        # next, we initialize the weights for each layer.
        
 
        
        X_arr = X.values
        
        scores = {'training_mse':[], 'validation_mse':[]}
        X_indices = list(range(X_arr.shape[0]))
        
        
        for _ in range(self.epochs):
            
            
            np.random.shuffle(X_indices)
            
            for point_index in X_indices:
            # let's first grab what the final output is supposed to be
                weight_updates = [np.zeros_like(layer) for layer in self.weights]
                bias_updates = [np.zeros_like(layer) for layer in self.biases]
                
                y_t = y.iloc[point_index]

                # propagate just one point for testing

                # get the outputs by propagating through
                outputs = self.propagate_(X_arr[point_index])

                # get the deltas through backprop
                deltas = self.backprop_(outputs, y_t)
                
                # initialize containers for point updates
                point_weight_updates = [np.zeros_like(layer) for layer in self.weights]
                point_bias_updates = [np.zeros_like(layer) for layer in self.biases]

                # calculate the weight updates
                point_weight_updates, point_bias_updates = self.calculate_updates_(deltas, outputs, X_arr[point_index])
                
                for i in range(len(weight_updates)):
                    weight_updates[i] += point_weight_updates[i]
                for i in range(len(bias_updates)):
                    bias_updates[i] += point_bias_updates[i][0]
                    
                # though the loss function is MSE, we're doing SGD. So we don't need to average any updates really!

                self.update_(weight_updates, bias_updates)
            # no need to do this if they dont want the plot -- computationally expensive
            if self.show_training_plot:
                training_mse= evaluate_reg(self.predict(X), y, 'mse')
                validation_mse= evaluate_reg(self.predict(validation_X), validation_y, 'mse')
                scores['training_mse'].append(training_mse)
                scores['validation_mse'].append(validation_mse)
        if self.show_training_plot:
            df = pd.DataFrame(scores )
            df.plot()

        
        
    def update_(self, weight_updates, bias_updates):
        """
        Helper function to update weights and biases
        
        Parameters:
        weight_updates : list of numpy arrays of updates for weights
        bias_updates : numpy array of updates for weights
        
        """
        for i in range(len(self.weights)):

            self.weights[i] += weight_updates[i]
            # i goofed the update somewhere and i can't find the error...
            self.biases[i] += bias_updates[i][0]


        
    def calculate_updates_(self, deltas, outputs, point):
        """
        Helper function to calculate weight changes. Uses the errors, outputs, and training point
        
        Parameters:
        -------
        deltas: list of numpy arrays of errors*derivatives at each node
        outputs: list of numpy arrays of outputs at each node
        point: array of single training point
        
        Returns:
        -------
        list of numpy arrays of weight updates
        
        """

        weight_updates = [np.zeros_like(layer) for layer in self.weights]
        bias_updates = [np.zeros_like(layer) for layer in self.biases]
        for layer_index, layer in enumerate(weight_updates):
            
            # we can calculate quite easily using the outer product instead of doing it one by one
            # the outputs of the previous layer are now the inputs of this current layer
            
            # the very first weight updates are done with the input example
            if layer_index == 0:
                weight_updates[layer_index] = self.learning_rate * np.outer(deltas[layer_index], point)
                bias_updates[layer_index] = self.learning_rate * deltas[layer_index] 
            else:

                weight_updates[layer_index] = self.learning_rate * np.outer(deltas[layer_index], outputs[layer_index - 1])
                bias_updates[layer_index] = self.learning_rate * deltas[layer_index] 
                
        return weight_updates, bias_updates
        
        

    def backprop_(self, outputs, y_t):
        """
        Helper function to calculate backpropagation. Uses the expected output and the outputs calculated for
        every node to backpropagage the error*derivatives, which is shorthanded as delta.
        
        Parameters:
        --------
        outputs : list of numpy arrays
        
        Returns:
        --------
        list of numpy arrays of deltas
        
        """
        
        # numpy doesn't support jagged arrays natively so we have to do this horrid thing. really
        # it's just so we have the same shape.
        deltas = [np.zeros_like(layer) for layer in outputs]

        for layer_index, output in reversed(list(enumerate(outputs))):
                      
            # the output layer is a special case since the error here is computed using the training example
            
            if layer_index == len(outputs) - 1:
                                
                output_error = np.array([y_t - output])
                deltas[layer_index] = output_error

            else:
            
            # for each node, the error (and hence delta) is taken as a weighted sum of the errors in the following layer.
            # since the weights for a layer are made in a N x M ndarray, where N corresponds to the 
            # number of errors of the following layer, which is given as a 1 x N ndarray, we can just 
            # perform matrix multiplication for the errors of the following layer and the weights.
            # then, we can just multiply it elementwise by the derivative of the activation function.
            # the errors*derivatives will backprop all in one step because it's easier to write it like this!
            
            # since we're doing sigmoid, the derivative would be: output * ( 1 - output)
            # NB : @ is the shorthand for numpy matrix multiplication
                derivative = outputs[layer_index] * (1 - outputs[layer_index])
                following_layer_deltas = deltas[layer_index + 1]
                weights_responsible = self.weights[layer_index + 1]
                deltas[layer_index] = (following_layer_deltas @ weights_responsible) * derivative
        
        return deltas
            
                
            
        
        
        
    def propagate_one_layer_(self, layer, bias, input_point):
        """
        Helper function to propagage one set of input point values through one layer of the network
        
        Parameters:
        ----
        Weights: Weight of the node to propagae point for
        
        """
        outputs = input_point @ layer.T + bias
        
        # we can replace self.sigmoid_ with a tanh or any other activation function
        vfunc = np.vectorize(self.sigmoid_)
                
        return vfunc(outputs)
    
    
    def propagate_(self, point):
        """
        Helper function to propagate one point through the entire network.
        
        Paramters:
        ----
        point: array of single training point
        
        
        Returns:
        -----
        list of numpy array of outputs at each node
        """
        
        outputs = []
        inputs = point
        
        # use the outputs of one layer as inputs into the next layer. continually propagate
        # the point through the network until we get to the end!!

        for layer_index, layer in enumerate(self.weights):
            if layer_index == len(self.weights) - 1:
                output = inputs @ layer.T + self.biases[layer_index]
                outputs.append(output)
            else:
                output = self.propagate_one_layer_(layer, self.biases[layer_index], inputs)
                # remember what the output is so we can use them for backprop
                outputs.append(output)
                inputs = output
        return outputs
                
        
    
    
    def predict(self, X):
        """
        Predict output of points given input X. 
        
        
        
        """
        if isinstance(X, pd.DataFrame):
            results = []
            for _, point in X.iterrows():
                results.append(self.predict(point))
            return pd.Series(results)
        
        
        return self.propagate_(X)[-1][0]
        
        
        
        
    def sigmoid_(self, o):
        """
        Helper sigmoid function.
        
        Parameters:
        -----
        o : input to the sigmoid function
        
        Returns:
        -----
        float of sigmoid function
        """
        return 1/ (1 + np.exp(-o))
        
    
    
    
    
    
    
    
    
    
    
    