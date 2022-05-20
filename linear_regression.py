from toolkit import evaluate_reg
import numpy as np
import pandas as pd



class LinearRegression:
    

    def __init__(self, learning_rate, epochs, show_training_plot = False):
        """
        Constructor for logistic regression algorithm 

        Parameters: 
        learning_rate : float of eta value to control learning rate or step size. 
        epochs: number of epochs to train the sample
        show_training_plot : whether or not to plot training/validation curves
        """
        # this is the only adjustable learning rate parameter. should be at or under 0.2
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.show_training_plot = show_training_plot

        # since each class has a different set of weights, let's try a hash map
        self.weights = None
        self.bias = None
        

        # we'll get these once we fit the data to the algo
        
        
    def fit(self, X, y, validation_X, validation_y):
        """
        Function to train model on data. 

        Parameters:
        X : pandas DataFrame of training data features
        y : pandas Series of training data
        validation_X : pandas DataFrame of validation data features
        validation_y : pandas Series of validation real outputs
        """

        
        # for each class i = 1,...K, 
        # initialize a weight vector for each of the features (number of columns in X)
        self.weights = np.random.default_rng().uniform(-0.01, 0.01, X.shape[1])
        
        # initialize a bias for each class
        self.bias = np.random.default_rng().uniform(-0.01, 0.01)
        
        # things will be faster if we use the np array representation instead of a DataFrame
        X_arr = X.values
        y_arr = y.values
        
        scores = {'training_mse':[], 'validation_mse':[]}
        for _ in range(self.epochs):
            # set the weight updates to 0

            updates = np.zeros(X_arr.shape[1])
            
            # let's get the starting weights so we can calculate how the weights change after
            # each runthrough of the dataset
            start_weights = self.weights
            
            weight_update = np.zeros(X_arr.shape[1])
            bias_update = 0

            
            # we could honestly get this code running a lot faster if just used 
            # nested arrays instead of dicts... oh well
            for index, x_t in enumerate(X_arr):
                
                # calculate the predicted output for each row of data
                
                output = np.dot(self.weights, x_t) + self.bias
                
                # calculate the error
                error = output - y_arr[index]
                # calculate updates for weights and bias
                weight_update +=  error * x_t
                bias_update +=  error
                
            # update weights and bias
            self.weights = self.weights - self.learning_rate * weight_update / (X_arr.shape[0])
            self.bias = self.bias - self.learning_rate * bias_update / (X_arr.shape[0])


            # grab the recalculated weights for each class
            end_weights = self.weights
            
            # get the abs value of the weight changes 
            weight_changes = abs(start_weights - end_weights)
            
            # calculate the scores for validation and testing
            if self.show_training_plot:
                training_mse= evaluate_reg(self.predict(X), y, 'mse')
                validation_mse= evaluate_reg(self.predict(validation_X), validation_y, 'mse')
                scores['training_mse'].append(training_mse)
                scores['validation_mse'].append(validation_mse)
        if self.show_training_plot:
            df = pd.DataFrame(scores)
            df.plot()
        
        


    
    def predict(self, X):
        
        """
        Predict a value (or multiple) using learned weights. Cannot be called if .fit() has not yet been called.
        
        Parameters:
        X : pandas Series or DataFrame of data to be predicted
        
        Returns:
        String or pandas Series of predicted classes
        
        """
        
        if isinstance(X, pd.DataFrame):
            return pd.Series([self.predict(x) for _, x in X.iterrows()])
                
        return np.dot(self.weights, X) + self.bias
    
    
    
    
    
    
    
    