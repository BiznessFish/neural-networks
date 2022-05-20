from toolkit import evaluate_cls
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm


class LogisticRegression:
    

    def __init__(self, learning_rate, epochs, show_training_plot = False):
        """
        Constructor for logistic regression algorithm 

        Parameters: 
        learning_rate : float of eta value to control learning rate or step size. 
        epochs: int of number of epochs used in training
        show_training_plot : bool of whether or not to plot training/validation curves
        """
        # this is the only adjustable learning rate parameter. should be at or under 0.2.
        # This is also arguably the most important tunable parameter for all machine learning models that
        # use gradient descent.
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.show_training_plot = show_training_plot

        # since each class has a different set of weights, let's try a hash map
        self.weights = None
        self.biases = None

        # we'll get these once we fit the data to the algo
        self.classes = None
        
        
    def fit(self, X, y, validation_X, validation_y):
        """
        Function to train model on data. 

        Parameters:
        X : pandas DataFrame of training data features
        y : pandas Series of training data 
        validation_X : pandas DataFrame of validation data features
        validation_y : pandas Series of validation real outputs
        """
        # grab unique classes from the target variable
        self.classes = y.unique()
        
        # for each class i = 1,...K, 
        # initialize a weight vector for each of the features (number of columns in X)
        self.weights = {i : np.random.default_rng().uniform(-0.01, 0.01, X.shape[1]) for i in self.classes}
        
        # initialize a bias for each class
        self.biases = {i : np.random.default_rng().uniform(-0.01, 0.01) for i in self.classes}
        
        # things will be faster if we use the np array representation instead of a DataFrame
        X_arr = X.values
        y_arr = y.values
        
        
        scores = {'training_accuracy':[], 'validation_accuracy':[]}
        for _ in range(self.epochs):
            # set the weight updates to 0
            # we don't need to do these separately (really, we can just extend the array)
            # but, for the sake of clarity, i'd like to!
            weight_updates = {i : np.zeros(X_arr.shape[1]) for i in self.classes}
            bias_updates = {i : 0 for i in self.classes}
            # we could honestly get this code running a lot faster if just used 
            # nested arrays instead of dicts... oh well
            for index, x_t in enumerate(X_arr):


                # calculate the outputs for each class to be converted to a probability
                # this for loop and the weighting below constitute softmax
                outputs = {i : 0 for i in self.classes}
                output_sum = 0
                for k, v in outputs.items():
                    # update the outputs for each class
                    output = np.dot(self.weights[k], x_t) + self.biases[k]
                    outputs[k] = np.exp(output)
                    output_sum += outputs[k]
                # convert the outputs to a probability by weighting it with softmax
                probabilities = {k : v/output_sum for k, v in outputs.items()}
                # calculate the weight and bias changes for each class. 
                # if the real class matches whatever outupt this data point has, it has a r_t of 1, 
                # if it doesn't, then it has an r_t of 0, since these are probabilities.
                for k, v in weight_updates.items():
                    if k != y_arr[index]:
                        weight_updates[k] += x_t*(-probabilities[k])
                        bias_updates[k] += -probabilities[k]
                    else:
                        weight_updates[k] += x_t*(1-probabilities[k])
                        bias_updates[k] += 1-probabilities[k]

                # update the weight parameters using the updates

            for k, v in self.weights.items():
                self.weights[k] += weight_updates[k]*self.learning_rate/X_arr.shape[0]
                self.biases[k] += bias_updates[k]*self.learning_rate/X_arr.shape[0]

            if self.show_training_plot:
                training_accuracy= evaluate_cls(self.predict(X), y, 'accuracy')
                validation_accuracy= evaluate_cls(self.predict(validation_X), validation_y, 'accuracy')
                scores['training_accuracy'].append(training_accuracy)
                scores['validation_accuracy'].append(validation_accuracy)
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
        
        # for a prediction, we use softmax to calculate the probability of each class and grab
        # the prediction with the highest probability.
        probabilities = {}
        o_k = 0
        for k, v in self.weights.items():
            prediction = np.dot(v, X) + self.biases[k]
            probabilities[k] = np.exp(prediction)
            o_k += probabilities[k]
        for k, v in probabilities.items():
            probabilities[k] = probabilities[k]/(o_k)
        return max(probabilities, key=probabilities.get)
    
    
    
    
    
    
    
    