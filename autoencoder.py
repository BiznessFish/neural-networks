import pandas as pd
import numpy as np
from toolkit import evaluate_reg

class AutoEncoder:
    

    def __init__(self, learning_rate, hidden_layer_size, epochs, show_training_plot = False):
        """
        Autoencoder for data. This attempts to perform nonlinear dimensionality reduction

        Parameters: 
        learning_rate : float of eta value to control learning rate or step size. 
        hidden_layer_size : int of number of nodes to use in hidden layer. Should generally be less than the number of features in the dataset
        epochs: int of number of epochs used in training
        show_training_plot : bool of whether or not to plot training/validation curves
        """
        self.learning_rate = learning_rate
        self.hidden_layer_size= hidden_layer_size
        self.epochs = epochs
        self.show_training_plot = show_training_plot

        self.weights = None
        self.biases = None

        
    def fit(self, X, validation_X):
        """
        Function to train model on data. 

        Parameters:
        X : pandas DataFrame of training data features
        y : pandas Series of training data 
        validation_X : pandas DataFrame of validation data features
        validation_y : pandas Series of validation real outputs
        """
        # grab dimensionality of data
        
        j = X.shape[1]
        
        self.weights = []
        self.biases = []
        
        # Make weights for the hidden layer and for the output layer
        self.weights.append( np.random.default_rng().uniform(-0.01, 0.01, (self.hidden_layer_size, j)) )
        self.weights.append( np.random.default_rng().uniform(-0.01, 0.01, (j, self.hidden_layer_size)) )
        
        self.biases.append( np.random.default_rng().uniform(-0.01, 0.01, self.hidden_layer_size) )
        self.biases.append( np.random.default_rng().uniform(-0.01, 0.01, j ) )      
        
        X_arr = X.values
        val_X_arr= validation_X.values
        X_indices = list(range(X_arr.shape[0]))
        scores = {'training_reconstruction_error':[], 'validation_reconstruction_error':[]}
        for _ in range(self.epochs):
            
            np.random.shuffle(X_indices)
                
            for point_index in X_indices:
                weight_updates = [np.zeros_like(layer) for layer in self.weights]
                bias_updates = [np.zeros_like(layer) for layer in self.biases]

                # get the outputs by propagating through
                outputs = self.propagate_(X_arr[point_index])

                # get the deltas through backprop
                deltas = self.backprop_(outputs, X_arr[point_index])
                
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
            
            if self.show_training_plot:

                average_training_reconstruction_error = 0
                average_validation_reconstruction_error = 0
                for i in range(len(X_arr)):
                    training_prediction = self.predict(X_arr[i])
                    average_training_reconstruction_error += np.linalg.norm(training_prediction - X_arr[i])
                for i in range(len(val_X_arr)):
                    validation_prediction = self.predict(val_X_arr[i])
                    average_validation_reconstruction_error += np.linalg.norm(validation_prediction - val_X_arr[i])
                average_training_reconstruction_error/= X_arr.shape[0]
                average_validation_reconstruction_error/=val_X_arr.shape[0]
                scores['training_reconstruction_error'].append(average_training_reconstruction_error)
                scores['validation_reconstruction_error'].append(average_validation_reconstruction_error)
#                 scores['validation_reconstruction_error'].append(validation_accuracy)
        if self.show_training_plot:
            df = pd.DataFrame(scores)
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
        
        
        return self.propagate_(X)[-1]
        
        
        
        
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
        
    
    