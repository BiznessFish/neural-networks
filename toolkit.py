# TODO: 
# - add more evaluation metrics
# - add error handling if calling evaluation metric on the wrong type of model 



import numpy as np
import pandas as pd

def generate_k_folds(y, num_folds = 5, classification=True):

    """Get indices of k-folds, equally sized, and distributed according to the target variable.
    
    y         : the column of the target variable
    num_folds : number of folds to make. Defaults to 5.
    --------
    
    Returns   : a nested array of indices. The i-th index of the array will correspond to a list of the indices of the k-th fold. 

    """
    
    folds = [[] for _ in range(num_folds)]
    
    if classification:
    
        lookup = {}

        for index, item in y.iteritems():
            if item not in lookup.keys():
                lookup[item] = 0
            fold_num = lookup[item]
            folds[fold_num].append(index)
            lookup[item] += 1
            if lookup[item] == num_folds:
                lookup[item] = 0
    else:
        count = 0
        for index, item in y.iteritems():
            folds[count % num_folds].append(index)
            count+= 1
        
    return folds





def impute_values_numeric(df_column, missing_val):
    """
Replaces instances of missing values with the average of that feature column
    
    
    df_column  : pandas Series with missing values to be imputed
    missing_val: missing value representation, usually '?' or NaN

    --------    
    returns : pandas Series with imputed values filled in
    """

    if isinstance(df_column, pd.core.series.Series):
        df_column = df_column.replace(missing_val, np.NaN)
        df_column = pd.to_numeric(df_column)
        df_column = df_column.fillna(df_column.mean())
        return df_column
    else:
        raise TypeError('Input must be a pandas Series')






def encode_one_hot(dataframe, feature, drop_original=True):
    """One hot encodes a column of a dataframe given the column name
    
    
    dataframe  : pandas dataframe
    feature: String of the name of the column to be one-hot encoded
    drop_original: whether or not to drop the original column

    --------    
    returns : pandas dataframe
            
    """


    if isinstance(dataframe, pd.core.frame.DataFrame):
        dummies = pd.get_dummies(dataframe[feature], prefix = feature)
        df = pd.concat([dataframe, dummies], axis = 1)
        if drop_original:
            df = df.drop(feature, axis = 1)
            return df
        else:
            return df
    else:
        raise TypeError('Input must be a pandas dataframe')

        
"""

TODO: add some dang error handling what is wrong with you


Ordinal encoder object that uses a given mapping to ordinally encode a data


"""



class OrdinalEncoder:
    
    
    """
    Fits a given mapping of an ordinal encoding of the given feature values. 
    
    Parameters:
    
    mappings: dict of mapping of feature values to ordinal values.
    
    """
    
    
    def fit(self, mappings):
        self.mappings = mappings
    
    
    """
    Transforms the target feature column using given ordinal mappings. 
    
    Parameters:
    
    mappings: dict of mapping to ordinal values. Value that appears in the feature columns are used as keys 
    feature_column: feature column from dataset that we want to encode. Does not necessarily have to be of datatype object
    ----
    Returns:
    
    Pandas Series of the transformed feature column that is ordinally encoded according to the given mapping
    
    
    """
    def fit_transform(self, mappings, feature_column):
        self.fit(mappings)
        return self.transform(feature_column)

    
    
    """
    Transforms the target feature column using saved ordinal mappings. 
    
    Parameters:
    
    feature_column: feature column from dataset that we want to encode. Does not necessarily have to be of datatype object
    ----
    Returns:
    
    Pandas Series of the transformed feature column that is ordinally encoded according to the saved mappings
    
    """
    def transform(self, feature_column):
        return feature_column.replace(self.mappings)
        
        

        
        
class Standardizer:
    """Standardizer for a pandas series."""

    
    
    def fit(self, column):
        """Takes a pandas Series and stores the mean and standard deviation
        
            column    : pandas Series. Must have numeric values.
        
        """
        
        if isinstance(column, pd.core.series.Series):
            if pd.api.types.is_numeric_dtype(column):
                self.mean = column.mean()
                self.std = column.std()
            else:
                raise TypeError('Needs to be numeric')
        else:
            raise TypeError('Must be pandas Series')

            
            

    def transform(self, column):
        """Takes a pandas Series and performs a z-scale transform on the elements within the series. Uses the mean and standard deviation stored in the object. Cannot be called if fit or fit_transform have not been called first. 
        
            column    : pandas Series
            -----
            returns    : z-scale transformed pandas Series
            
            """
        if self.mean == None:
            raise Error('Must first call fit or fit_transform')
        
        if isinstance(column, pd.core.series.Series):
            if pd.api.types.is_numeric_dtype(column):
                column = (column - self.mean)/self.std
                return column
            else:
                raise TypeError('Needs to be numeric')
        else:
            raise TypeError('Must be pandas Series')

            

    def fit_transform(self, column):
        """Wrapper function for fit and transform. Calls fit then transform.
             column    : pandas Series
             -----
             returns    : z-scale transformed pandas Series
        
        """
        self.fit(column)
        column = self.transform(column)
        return column
        
            
            

def evaluate_reg(predicted, true_value, metric):
    
    """Given two pandas Series of predicted values and true values, calculates an evaluation metric for regression tasks.

    predicted    : pandas Series of predicted values
    true_value   : pandas Series of actual values
    metric       : String of evaluation metric chosen, can be 
                 'r_squared', 'mse'

    --------
    returns float of the requested metric

    """
    
    
    
    if isinstance(predicted, pd.core.series.Series) and isinstance(true_value, pd.core.series.Series):            
        
        if len(predicted) == len(true_value):

        
            if metric == 'r_squared':

                if pd.api.types.is_numeric_dtype(predicted) and pd.api.types.is_numeric_dtype(true_value):

                    residuals = predicted.values - true_value.values
                    SS_res = (residuals**2).sum()
                    SS_tot = ((true_value - true_value.mean())**2).sum()
                    r_squared = 1 - SS_res/SS_tot
                    return r_squared

                else:

                    raise TypeError('pandas Series must contain numeric datatype for r_squared')


            if metric == 'mse':

                if pd.api.types.is_numeric_dtype(predicted) and pd.api.types.is_numeric_dtype(true_value):

                    predicted, true = predicted.values, true_value.values
                    mse = np.mean((predicted-true)**2)

                    return mse
                    
                    
        else:
            
            raise ValueError('Predicted values and true values must be the same length!')

                
    else: 
        raise TypeError('Can only pass pandas Series to this function')
    
    
        

def evaluate_cls(predicted, true_value, metric):
    """Given two pandas Series of predicted values and true values, calculates an evaluation metric for classification tasks.

    predicted    : pandas Series of predicted values
    true_value   : pandas Series of actual values
    metric       : String of evaluation metric chosen, can be 
                   'accuracy', 'f1'

    --------
    returns float of the requested metric

    """

    if isinstance(predicted, pd.core.series.Series) and isinstance(true_value, pd.core.series.Series):
                
        if len(predicted) == len(true_value):
                        
            if metric == 'accuracy':
                
                return (predicted.reset_index(drop=True) == true_value.reset_index(drop=True)).sum()/len(predicted)
            
            
            if metric == 'f1':
                # not sure what'll happen if we happen to run into a case where we predict something that isn't in the real values...
                num_classes = true_value.nunique()
                predicted = predicted.values
                true_value = true_value.values
                classes = np.unique(true_value)

                f1 = 0 
                for cat in classes: 
                    # use the bitwise and here
                    true_positive = np.sum((true_value == cat) & (predicted == cat))
                    false_positive = np.sum((true_value != cat) & (predicted == cat))
                    false_negative = np.sum((true_value == cat) & (predicted != cat))
                    # you can run into situations where true pos/false pos/false neg are all 0.... not sure what to do about that..
                    # for now we can just set everything to 0 if thats the case
                    if true_positive == 0 and false_positive == 0:
                        precision = 0
                    else:
                        precision = true_positive/(true_positive + false_positive)
                    if true_positive == 0 and false_negative == 0:
                        recall = 0
                    else:
                        recall = true_positive/(true_positive + false_negative)
                        
                    if precision == 0 and recall == 0:
                        f1 = 0
                    else:
                        f1 += 2 * (precision * recall) / (precision + recall)
                return f1/num_classes
            
            if metric == 'cross_entropy':
                
                # first get the proportions
                true_props = true_value.value_counts(normalize = True)
                predicted_props = predicted.value_counts(normalize = True)

                if set(predicted_props.index) == set(true_props.index):
                    # grab the numpy representation of the Series sorted by index (class label), effectively match by class label by array index.
                    predicted_props = predicted_props.sort_index().values 
                    true_props = true_props.sort_index().values
                    
                    # do the calc
                    predicted_log = np.log2(predicted_props)
                    dot = np.dot(true_props, predicted_log)
                    return -dot

                else:
                # a bit of handling for edge cases; what if elements of one aren't in the other? 

                    missing_items = set(true_props.index) - set(predicted_props.index)

                    if missing_items:
                        new_items = pd.Series({ele : 0 for ele in missing_items})
                        predicted_props = predicted_props.append(new_items)

                    missing_items = set(predicted_props.index) - set(true_props.index)

                    if missing_items:
                        new_items = pd.Series({ele : 0 for ele in missing_items})
                        true_props = true_props.append(new_items)

                    # now that we have the missing elements added, do the calculation as above
                    predicted_props = predicted_props.sort_index().values 
                    true_props = true_props.sort_index().values
                    predicted_log = np.log2(predicted_props)
                    dot = np.dot(true_props, predicted_log)
                    return -dot
        
        

       
        
                    
        else:
            
            raise Error('Predicted values and true values must be the same length!')

    else:
        
        raise TypeError('predicted and true values must be pandas Series')
    



# TODO: Write docstrings for algorithms, 
    
    
class MajorityClassifier:
    

    def fit(self, X, y):

        self.pred = y.mode()[0]


    def predict(self, X):

        return pd.Series([self.pred for _ in range(len(X))])
    
    
    def score(self, X, y, metric = 'accuracy'):
        
        predicted = self.predict(X)
                
        
        return evaluate_cls(predicted, y, metric)
        



class MeanRegressor:
    
    def fit(self, X, y):
        
        self.pred = y.mean()
        
        
    def predict(self, X):
        
        return pd.Series([self.pred for _ in range(len(X))])

        
    def score(self, X, y, metric = 'mse'):
        
        predicted = self.predict(X)
                
        
        return evaluate_reg(predicted, y, metric)


    
    
# TODO: kwargs for knn algorithms? they won't fit with this k-fold CV function as it is... no way to change k     
    
    
"""
Peforms k-fold cross validation using given algorithm. Each fold of k-folds will rotate acting as the testing set, while the rest of the folds will act as the training set. For classification tasks, will attempt to match the distribution of the target variable as evenly as possible across the k-folds.


Parameters:

model          : Name of model Object to call .fit() on
X              : pandas Dataframe of feature variables.
y              : pandas Series Target variables.
metric         : String of evaluation metric to score performance on (things like 'mse', 'accuracy' etc.)
num_folds      : int of number of folds to perform cross validation over.
verbose        : If True, will output the requested evaluation metric for each individual fold. Defaults to False.
classification : If True, will attempt to select folds in which the target distribution is kept as similarly as      possible across folds. Defaults to False.
scale          : Dict of which features need to be scaled and by which method. Keys should be feature names and values should be scaling methods. Can be 'standard', 'min_max'. Scaling of this format will be fit on whichever folds are the training set. Scaling of the testing set will use the learned values from the training set.

------

Returns:

Returns the requested metric averaged over k.
    

"""    
    
def cross_validate(model, X, y, metric, num_folds = 5, verbose = False, classification=False, scale = None):
    
    if not isinstance(X, pd.core.frame.DataFrame):
        raise TypeError("Data must be in a pandas dataframe")
    if not isinstance(y, pd.core.series.Series):
        raise TypeError("target must be a pandas Series")
    if not isinstance(normalize, list) and normalize is not None:
        raise TypeError("Normalize must be a list")
    if scale is not None:
        if not set(scale.keys()).issubset(set(X.columns)):
            raise NameError("Keys of scale must contain column names of dataset")
    
    
    indices = generate_k_folds(y = y, num_folds = num_folds, classification = classification)
    scores = []
    
    for fold in range(num_folds):
        test_indices = indices[fold]
        
        test_X = X.loc[test_indices]
        train_X = X.loc[X.index.difference(test_indices)]
        
        test_y = y.loc[test_indices]
        train_y = y.loc[y.index.difference(test_indices)]
        
        if scale is not None:
            
            
            # For each feature, use the method of scaling specified. If it is standard scaling,
            # instantiate a Standardizer and fit the specified column. If it is min_max, do that, and so on. 
            for feature, method in scale:
                if method == 'standard':
                    standardizer = Standardizer()
                    train_X[feature] = standardizer.fit_transform(train_X[feature])
                    test_X[feature] = standardizer.transform(test_X[feature])
                    
                
                # Since we haven't implemented min_max scaling yet....just do nothing
                elif method == 'min_max':
                    continue
                    
                else:
                    raise InputError(f"Scaling method {method} given to function not recognized.")
    
        
        model.fit(X = train_X, y = train_y)
        score = model.score(X = test_X, y = test_y, metric = metric)
        
        fold_size = len(test_indices)
        
        if verbose:
            print(f"{metric} for Fold {fold} as test is {score}")
            print(f"Fold {fold} has {fold_size} elements")
        
        scores.append(score)
        
    average = np.mean(scores)
    print(f"Average score for {num_folds}-fold cross validation is {average}")
    return average



"""
Kernel function to return the weight for each observation point based on given smoothing parameter sigma. 
Sigma defaults to 0.5

Parameters:

distance : numeric value of a distance to be kernelized
smoothing_parameter : adjustable smoothing parameter for how wide the kernel is.

------
Returns:



"""

# TODO : add in quick check for numeric datatype for input distance....

def gaussian_kernel(distance, smoothing_parameter = 0.5): 
    return np.e**((-1/(2*smoothing_parameter))*distance)


    

    
    
class KNNRegressor:
    
    """
    A KNN model. Memorizes testing data and makes predictions on a regression problem using the KNN rule. Regression prediction will be weighted using a RBF kernel with an adjustable smoothing paramter sigma. Does not take non numeric features. All distance calculations will be calculated with Euclidean distance.
    
    
    Atttributes:
    
    k_neighbors : Number of neighbors to consider during the prediction algorithm. Defaults to 3.
    sigma : Width of the smoothing parameter for RBF. Defaults to 0.5.
    
             
    """
    
    
    def __init__(self, k_neighbors = 3, sigma = 0.5, verbose = False):
        if k_neighbors == 0 or k_neighbors < 0:
            raise InputError("k must be at least 1 and non-negative")
        self.k_neighbors = k_neighbors
        self.sigma = sigma
        self.verbose = verbose
        
    
    
    # Fitting this model is really just... storing the training data.
    # Since this algorithm doesn't do any work until query time
    
    def fit(self, X, y):
        if not isinstance(X, pd.core.frame.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame")
        if not isinstance(y, pd.core.series.Series):
            raise TypeErro('Input target must be a pandas Series')
        if len(X) != len(y):
            raise ValueError('Features and target variables must have matching length')
        if not pd.api.types.is_numeric_dtype(y):
            raise TypeError('Target variable must be numeric data type')
        for column, dtype in X.dtypes.iteritems():
            if not pd.api.types.is_numeric_dtype(dtype):
                raise TypeError(f'{column} is not a numeric data type')        
            
        self.y = y
        self.X = X
        
        
    """
    This function predicts points based on the learned dataset. Will default to Euclidean distance (L2 norm) if no features were marked as categorical. If all features are marked as categorical, distance calculations will be a VDM. If some but not all are marked as categorical, a mixed type will be used.
    
    Parameters:
    
    X : Pandas Dataframe of points to be predicted.

    ------
    Returns:
    A pandas Series of predicted points.
   
    
    """
    def predict(self, X):

                
        if isinstance(X, pd.core.frame.DataFrame):
            for column, dtype in X.dtypes.iteritems():
                if not pd.api.types.is_numeric_dtype(dtype):
                    raise TypeError(f'{column} is not a numeric data type')



            # Get the indices of the k_neighbors for each point in the sample we want to predict
            # We want to keep track of the distance calculations for each point
            k_neighbors_distances = {}


            # This is probably the slowest implementation in history, but here we go:

            predictions = []



            for index, row in X.iterrows():


                distances = self.X.apply(lambda x : np.linalg.norm(row - x), axis = 1)

                # --------------------------------------------
                # Sort by this distance and find out the k closest ones.
                # If k is greater than the actual number of training examples we have, then just use all of them.
                if len(self.X) < self.k_neighbors:
                    neighbors = distances.sort_values()
                else:
                    neighbors = distances.sort_values()[:self.k_neighbors]


                prediction = None


                neighbors_output = self.y.loc[neighbors.index]
                



                # Kernelizing uses the distance metric, which is to be stored in neighbors
                kernel_weights = [gaussian_kernel(neighbor, self.sigma) for neighbor in neighbors]

                # In the event that the kernel weights are all 0, meaning that the distances for the nearest k neighbors are 
                # too far away to make a difference, or sigma is too small, we can simply return the mean of the closest candidates
                if sum(kernel_weights) == 0 :
                    predictions = [neighbors_output.mean() for _ in neighbors_output]

                else :
                    prediction = sum(kernel_weights[_] * neighbors_output.iloc[_] for _ in range(len(neighbors_output)) ) / sum(kernel_weights)


                predictions.append(prediction)


                # -----------------

            # We can just return the predictions now as a list, or we can add the indices.

            # I think it makes sense to add the indices (?).

            return pd.Series(predictions, index = X.index)

        elif isinstance(X, pd.core.series.Series) :
            
            distances = self.X.apply(lambda x : np.linalg.norm(X - x), axis = 1)

            # --------------------------------------------
            # Sort by this distance and find out the k closest ones.
            # If k is greater than the actual number of training examples we have, then just use all of them.
            if len(self.X) < self.k_neighbors:
                neighbors = distances.sort_values()
            else:
                neighbors = distances.sort_values()[:self.k_neighbors]


            prediction = None


            neighbors_output = self.y.loc[neighbors.index]
            
            if self.verbose:
                print("Neighbors for this point are: ", neighbors_output, "with features", self.X.loc[neighbors.index])

            # Kernelizing uses the distance metric, which is to be stored in neighbors
            kernel_weights = [gaussian_kernel(neighbor, self.sigma) for neighbor in neighbors]

            # In the event that the kernel weights are all 0, meaning that the distances for the nearest k neighbors are 
            # too far away to make a difference, or sigma is too small, we can simply return the mean of the closest candidates
            if sum(kernel_weights) == 0 :
                predictions = neighbors_output.mean()
            else :
                prediction = sum(kernel_weights[_] * neighbors_output.iloc[_] for _ in range(len(neighbors_output)) ) / sum(kernel_weights)


            return prediction
            

        
"""
A k-Nearest Neighbors classifier for classification tasks. For best resuls, make k_neighbors a prime number to guarantee that there are no ties. Generally, having k be an odd number is fine. For even k, we will randomly choose a number.

"""        
        
        
        
class KNNClassifier():
    
    def __init__(self, k_neighbors = 3, verbose = False):
        self.k_neighbors = k_neighbors
        self.verbose = verbose

        

        
    def fit(self, X, y):
        if not isinstance(X, pd.core.frame.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame")
        if not isinstance(y, pd.core.series.Series):
            raise TypeErro('Input target must be a pandas Series')
        if len(X) != len(y):
            raise ValueError('Features and target variables must have matching length')
        for column, dtype in X.dtypes.iteritems():
            if not pd.api.types.is_numeric_dtype(dtype):
                raise TypeError(f'{column} is not a numeric data type')         
        self.X = X
        self.y = y 
        
    
    def predict(self, X):
        if isinstance(X, pd.core.frame.DataFrame):
            for column, dtype in X.dtypes.iteritems():
                if not pd.api.types.is_numeric_dtype(dtype):
                    raise TypeError(f'{column} is not a numeric data type')



            # Get the indices of the k_neighbors for each point in the sample we want to predict
            # We want to keep track of the distance calculations for each point
            k_neighbors_distances = {}


            # This is probably the slowest implementation in history, but here we go:

            predictions = []



            for index, row in X.iterrows():


                distances = self.X.apply(lambda x : np.linalg.norm(row - x), axis = 1)

                # --------------------------------------------
                # Sort by this distance and find out the k closest ones.
                # If k is greater than the actual number of training examples we have, then just use all of them.
                if len(self.X) < self.k_neighbors:
                    neighbors = distances.sort_values()
                else:
                    neighbors = distances.sort_values()[:self.k_neighbors]

                # We get the mode, but sometimes there is more than one mode, so we randomly sample and take whatever we get.
                # If it's only one mode, then we get only one.

                prediction = self.y.loc[neighbors.index].mode().sample().iloc[0]



                predictions.append(prediction)


                # -----------------

            # We can just return the predictions now as a list, or we can add the indices.

            # I think it makes sense to add the indices (?).

            return pd.Series(predictions, index = X.index)

        elif isinstance(X, pd.core.series.Series) :
            
            distances = self.X.apply(lambda x : np.linalg.norm(X - x), axis = 1)

            # --------------------------------------------
            # Sort by this distance and find out the k closest ones.
            # If k is greater than the actual number of training examples we have, then just use all of them.
            if len(self.X) < self.k_neighbors:
                neighbors = distances.sort_values()
            else:
                neighbors = distances.sort_values()[:self.k_neighbors]
            


            prediction = None


            neighbors_output = self.y.loc[neighbors.index]
            
            if self.verbose:
                print("Neighbors for this point are: ", neighbors_output, "with features", self.X.loc[neighbors.index])
            
            prediction = self.y.loc[neighbors.index].mode().sample().iloc[0]
            return prediction
    
    
    
    
    def score(self, X, y, metric = 'mse'):
        
        predicted = self.predict(X)
                
        
        return evaluate_cls(predicted, y, metric)

        
            


"""
Edits the dataset based on the CNN rule. Returns a minimally consistent subset for Regression based tasks.

Parameters:

k_neighbors : int of number of neighbors to consider for regression
sigma : float of smoothing parameter

"""
class CondensedNearestNeighborRgrs:
    
    
   
    def __init__(self, k_neighbors, sigma, epsilon, verbose = False):
        self.k_neighbors = k_neighbors
        self.sigma = sigma
        self.epsilon = epsilon
        self.verbose = verbose
        
    """
    Called to train a minimally consistent subset using the CNN rule. Requires a k number for Nearest Neighbor decision rule. 
    
    Parameters
    X : pandas DataFrame of training set
    y : pandas Series of training set
    epsilon : float of allowable error metric 
            
    
    """
        
    def fit(self, X, y):
        store = []
        grabbag = []
        indices = list(X.index)
        
        # First pass
        
        # Start by putting one element of indices into store.
        
        store.append(indices.pop())
        
        knn_regressor = KNNRegressor(k_neighbors = 1, sigma = self.sigma)
        
        # Next, we bin the indices into either grabbag or store. store contains the elements that help us classify with the decision boundary
        
        for index in indices:
            knn_regressor.fit(X.loc[store], y.loc[store])
            
            # If the prediction is outside the error tolerance, then we can count that as an error
            print(y.loc[index])
            print(knn_regressor.predict(X.loc[index]))
            if abs(y.loc[index] - knn_regressor.predict(X.loc[index])) <= self.epsilon:
                if verbose:
                    print("Prediction is outside of error tolerance, adding point at index ", index)
                grabbag.append(index)
            else:
                store.append(index)
                
        # After first pass, we go through the grabbag. 
        # Depending on what elements are in store, some elements in grabbag may change the decision boundaries
        # We will now go through grabbag until it's empty or until nothing changes. 
        
        is_grabbag_change = True
        is_metric_better = True
        
        while grabbag and is_grabbag_change:
            
            # It's bad practice to modify a list you're iterating over, 
            # so we'll do the loop in this way instead of a regular for loop.
            bag_index = 0
            starting_length = len(grabbag)

            while bag_index < len(grabbag):
                
 
                # Refit the classifier using the indices in store
                knn_regressor.fit(X.loc[store], y.loc[store])

                # If the example is misclassified, then it helps with the decision boundary.
            
                if abs(y.loc[grabbag[bag_index]] - knn_regressor.predict(X.loc[grabbag[bag_index]])) > self.epsilon:
                    # Since we're popping at the index, we'll need to pick up where we left off, so we
                    # don't increment the index.
                    if verbose:
                        print("Prediction is outside of error tolerance, adding point at index ", index)
                    store.append(grabbag.pop(bag_index))
                else:
                    bag_index += 1
                    
                
            # If we've gone through the bag and gotten no change in the grabbag, then we kick out of the loop
            
            if len(grabbag) == starting_length:
                is_grabbag_change = False
        
        # At the end of this procedure, the grabbag will contain all of the examples that need to be condensed out of the given samples
        # store will contain all of the relevant points that help define the decision boundary.

        self.condensed_X = X.loc[store]
        self.condensed_y = y.loc[store]
        

        
    """
    This function predicts points based on the learned dataset. Will default to Euclidean distance (L2 norm) if no features were marked as categorical. If all features are marked as categorical, distance calculations will be a VDM. If some but not all are marked as categorical, a mixed type will be used.
    
    Parameters:
    
    X : Pandas Dataframe of points to be predicted.

    ------
    Returns:
    A pandas Series of predicted points.
   
    
    """
    def predict(self, X):
        if not isinstance(X, pd.core.frame.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame")
        for column, dtype in X.dtypes.iteritems():
            if not pd.api.types.is_numeric_dtype(dtype):
                raise TypeError(f'{column} is not a numeric data type')
                
        # Get the indices of the k_neighbors for each point in the sample we want to predict
        # We want to keep track of the distance calculations for each point
        k_neighbors_distances = {}
        
        
        # This is probably the slowest implementation in history, but here we go:
        
        predictions = []
        
        
    
        for index, row in X.iterrows():
            
            
            distances = self.condensed_X.apply(lambda x : np.linalg.norm(row - x), axis = 1)
            
            # --------------------------------------------
            # Sort by this distance and find out the k closest ones.
            # If k is greater than the actual number of training examples we have, then just use all of them.
            if len(self.condensed_X) < self.k_neighbors:
                neighbors = distances.sort_values()
            else:
                neighbors = distances.sort_values()[:self.k_neighbors]

            
            prediction = None
            

            neighbors_output = self.condensed_y.loc[neighbors.index]

            # Kernelizing uses the distance metric, which is to be stored in neighbors
            kernel_weights = [gaussian_kernel(neighbor, self.sigma) for neighbor in neighbors]

            # In the event that the kernel weights are all 0, meaning that the distances for the nearest k neighbors are 
            # too far away to make a difference, or sigma is too small, we can simply return the mean of the closest candidates
            if sum(kernel_weights) == 0 :
                predictions = neighbors_output.mean()
            else :
                prediction = sum(kernel_weights[_] * neighbors_output.iloc[_] for _ in range(len(neighbors_output)) ) / sum(kernel_weights)

                
            predictions.append(prediction)
            
            
            # -----------------
                    
        # We can just return the predictions now as a list, or we can add the indices.
        
        # I think it makes sense to add the indices (?).
        
        return pd.Series(predictions, index = X.index)
        
            


"""
Edits the dataset based on the CNN rule. Returns a minimally consistent subset for Regression based tasks.

Parameters:

k_neighbors : int of number of neighbors to consider for regression


"""
class CondensedNearestNeighborCls:
    
    
   
    def __init__(self, k_neighbors, verbose=False):
        self.k_neighbors = k_neighbors
        self.verbose = True

        
    """
    Called to train a minimally consistent subset using the CNN rule. Requires a k number for Nearest Neighbor decision rule. 
    
    Parameters
    X : pandas DataFrame of training set
    y : pandas Series of training set

            
    
    """
        
    def fit(self, X, y):
        store = []
        grabbag = []
        indices = list(X.index)
        
        # First pass
        
        # Start by putting one element of indices into store.
        
        store.append(indices.pop())
        
        knn_classifier = KNNClassifier(k_neighbors = 1)
        
        # Next, we bin the indices into either grabbag or store. store contains the elements that help us classify with the decision boundary
        
        for index in indices:
            knn_classifier.fit(X.loc[store], y.loc[store])
            
            # If the prediction is wrong, then we can count that as an error
            if y.loc[index] != knn_classifier.predict(X.loc[index]):
                if self.verbose:
                    print("Adding point at index ", index)
                grabbag.append(index)
            else:
                store.append(index)
                
        # After first pass, we go through the grabbag. 
        # Depending on what elements are in store, some elements in grabbag may change the decision boundaries
        # We will now go through grabbag until it's empty or until nothing changes. 
        
        is_grabbag_change = True
        is_metric_better = True
        
        while grabbag and is_grabbag_change:
            
            # It's bad practice to modify a list you're iterating over, 
            # so we'll do the loop in this way instead of a regular for loop.
            bag_index = 0
            starting_length = len(grabbag)

            while bag_index < len(grabbag):
                
 
                # Refit the classifier using the indices in store
                knn_classifier.fit(X.loc[store], y.loc[store])

                # If the example is misclassified, then it helps with the decision boundary.
            
                if y.loc[grabbag[bag_index]] != knn_classifier.predict(X.loc[grabbag[bag_index]]):
                    # Since we're popping at the index, we'll need to pick up where we left off, so we
                    # don't increment the index.
                    store.append(grabbag.pop(bag_index))
                else:
                    bag_index += 1
                    
                
            # If we've gone through the bag and gotten no change in the grabbag, then we kick out of the loop
            
            if len(grabbag) == starting_length:
                is_grabbag_change = False
        
        # At the end of this procedure, the grabbag will contain all of the examples that need to be condensed out of the given samples
        # store will contain all of the relevant points that help define the decision boundary.

        self.X = X.loc[store]
        self.y = y.loc[store]
        

        
    """
    This function predicts points based on the learned dataset. Will default to Euclidean distance (L2 norm) if no features were marked as categorical. If all features are marked as categorical, distance calculations will be a VDM. If some but not all are marked as categorical, a mixed type will be used.
    
    Parameters:
    
    X : Pandas Dataframe of points to be predicted.

    ------
    Returns:
    A pandas Series of predicted points.
   
    
    """
    def predict(self, X):
        if isinstance(X, pd.core.frame.DataFrame):
            for column, dtype in X.dtypes.iteritems():
                if not pd.api.types.is_numeric_dtype(dtype):
                    raise TypeError(f'{column} is not a numeric data type')



            # Get the indices of the k_neighbors for each point in the sample we want to predict
            # We want to keep track of the distance calculations for each point
            k_neighbors_distances = {}


            # This is probably the slowest implementation in history, but here we go:

            predictions = []



            for index, row in X.iterrows():


                distances = self.X.apply(lambda x : np.linalg.norm(row - x), axis = 1)

                # --------------------------------------------
                # Sort by this distance and find out the k closest ones.
                # If k is greater than the actual number of training examples we have, then just use all of them.
                if len(self.X) < self.k_neighbors:
                    neighbors = distances.sort_values()
                else:
                    neighbors = distances.sort_values()[:self.k_neighbors]

                # We get the mode, but sometimes there is more than one mode, so we randomly sample and take whatever we get.
                # If it's only one mode, then we get only one.

                prediction = self.y.loc[neighbors.index].mode().sample().iloc[0]



                predictions.append(prediction)


                # -----------------

            # We can just return the predictions now as a list, or we can add the indices.

            # I think it makes sense to add the indices (?).

            return pd.Series(predictions, index = X.index)

        elif isinstance(X, pd.core.series.Series) :
            
            distances = self.X.apply(lambda x : np.linalg.norm(X - x), axis = 1)

            # --------------------------------------------
            # Sort by this distance and find out the k closest ones.
            # If k is greater than the actual number of training examples we have, then just use all of them.
            if len(self.X) < self.k_neighbors:
                neighbors = distances.sort_values()
            else:
                neighbors = distances.sort_values()[:self.k_neighbors]
            


            prediction = None


            neighbors_output = self.y.loc[neighbors.index]
            
            if self.verbose:
                print("Neighbors for this point are: ", neighbors_output, "with features", self.X.loc[neighbors.index])
            
            prediction = self.y.loc[neighbors.index].mode().sample().iloc[0]
            return prediction
        
        
"""
Edits the dataset based on the ENN rule. If a training sample is erroneously estimated by the rest of the training data, it is discarded. This regresses using the KNN rule.

Parameters:

k_neighbors : int of number of neighbors to consider for regression. Defaults to 3.
sigma : float of smoothing parameter. Defaults to 0.5
epsilon : float of allowable error. Defaults to 100.
verbose : bool of whether to display points marked for deletion

"""
class EditedNearestNeighborRgrs:
    
    
   
    def __init__(self, k_neighbors = 3, sigma = 0.5, epsilon = None, verbose=False):
        self.k_neighbors = k_neighbors
        self.sigma = sigma
        self.epsilon = epsilon
        self.verbose = verbose
    """
    Called to train an edited dataset Requires a k number for Nearest Neighbor decision rule. 
    
    Parameters
    X : pandas DataFrame of training set
    y : pandas Series of training set target variable
    val_X : pandas DataFrame of validation set
    val_y : pandas Series of validation set target variable
    metric : evaluation metric for comparing performance
            
    
    """
        
    def fit(self, X, y, val_X, val_y, metric):
        self.X = X
        self.y = y
        
        #First get eval metric on validation set
        knn_regressor = KNNRegressor(self.k_neighbors, self.sigma)
        knn_regressor.fit(X, y)
        val_result = knn_regressor.predict(val_X)
        val_metric = evaluate_reg(val_result, val_y, metric)
        
        iteration = 1
        while True:   
            
            # Batch removal is easiest
            indices = list(self.X.index)        
            marked_indices = []

            for index in indices:
                EKNN_indices = list(indices)
                EKNN_indices.remove(index)

                knn_regressor = KNNRegressor(self.k_neighbors, self.sigma)
                knn_regressor.fit(self.X.loc[EKNN_indices], self.y.loc[EKNN_indices])

                # If the prediction is outside the error tolerance, then we can count that as an error
                if abs(self.y.loc[index] - knn_regressor.predict(self.X.loc[index])) >= self.epsilon:
                    marked_indices.append(index)

            
            # If no examples have been marked, we break the loop
            if not marked_indices:
                if self.verbose:
                    print("No points marked for batch removal. Fit has completed")
                break
            if self.verbose:
                print("Points marked for batch removal at indices: ", marked_indices)
            # Make a new list of indices so we can try out our eval metric
            new_indices = []
            

            for index in indices:
                if index not in marked_indices:
                    new_indices.append(index)

            # If new_indices is empty, then that means every point was misclassified
            
            if not new_indices:
                raise ValueError("Try raising epsilon. All points are misclassified on first run.")
                    
            knn_regressor.fit(self.X.loc[new_indices], self.y.loc[new_indices])
            new_val_result = knn_regressor.predict(val_X)
            new_val_metric = evaluate_reg(new_val_result, val_y, metric)
            
            # If the new indices are not worse, refit the data. Else, break.
            if new_val_metric <= val_metric:
                self.X = self.X.loc[new_indices]
                self.y = self.y.loc[new_indices]
                if self.verbose:
                    print(f"Improvement in {metric} detected. \n Old : {val_metric} \n New : {new_val_metric}")
            else:
                break
            iteration += 1
        
 
        
    """
    This function predicts points based on the learned dataset. Will default to Euclidean distance (L2 norm) if no features were marked as categorical. If all features are marked as categorical, distance calculations will be a VDM. If some but not all are marked as categorical, a mixed type will be used.
    
    Parameters:
    
    X : Pandas Dataframe of points to be predicted.

    ------
    Returns:
    A pandas Series of predicted points.
   
    
    """
    def predict(self, X):
        if not isinstance(X, pd.core.frame.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame")
        for column, dtype in X.dtypes.iteritems():
            if not pd.api.types.is_numeric_dtype(dtype):
                raise TypeError(f'{column} is not a numeric data type')
                
        # Get the indices of the k_neighbors for each point in the sample we want to predict
        # We want to keep track of the distance calculations for each point
        k_neighbors_distances = {}
        
        
        # This is probably the slowest implementation in history, but here we go:
        
        predictions = []
        
        
    
        for index, row in X.iterrows():
            
            
            distances = self.X.apply(lambda x : np.linalg.norm(row - x), axis = 1)
            
            # --------------------------------------------
            # Sort by this distance and find out the k closest ones.
            # If k is greater than the actual number of training examples we have, then just use all of them.
            if len(self.X) < self.k_neighbors:
                neighbors = distances.sort_values()
            else:
                neighbors = distances.sort_values()[:self.k_neighbors]

            
            prediction = None
            

            neighbors_output = self.y.loc[neighbors.index]

            # Kernelizing uses the distance metric, which is to be stored in neighbors
            kernel_weights = [gaussian_kernel(neighbor, self.sigma) for neighbor in neighbors]

            # In the event that the kernel weights are all 0, meaning that the distances for the nearest k neighbors are 
            # too far away to make a difference, or sigma is too small, we can simply return the mean of the closest candidates
            if sum(kernel_weights) == 0 :
                predictions = neighbors_output.mean()
            else :
                prediction = sum(kernel_weights[_] * neighbors_output.iloc[_] for _ in range(len(neighbors_output)) ) / sum(kernel_weights)

                
            predictions.append(prediction)
            
            
            # -----------------
                    
        # We can just return the predictions now as a list, or we can add the indices.
        
        # I think it makes sense to add the indices (?).
        
        return pd.Series(predictions, index = X.index)
        

        
"""
Edits the dataset based on the CNN rule. Returns a minimally consistent subset for classification based tasks.

Parameters:

k_neighbors : int of number of neighbors to consider for classification


"""
class EditedNearestNeighborCls:
    
    
   
    def __init__(self, k_neighbors, verbose=False):
        self.k_neighbors = k_neighbors
        self.verbose = verbose
    """
    Called to train a minimally consistent subset using the CNN rule. Requires a k number for Nearest Neighbor decision rule. 
    
    Parameters
    X : pandas DataFrame of training set
    y : pandas Series of training set target variable
    val_X : pandas DataFrame of validation set
    val_y : pandas Series of validation set target variable
    metric : evaluation metric for comparing performance
            
    
    """
        
    def fit(self, X, y, val_X, val_y, metric):
        self.X = X
        self.y = y
        
        #First get eval metric on validation set
        knn_classifier = KNNClassifier(self.k_neighbors)
        knn_classifier.fit(X, y)
        val_result = knn_classifier.predict(val_X)
        val_metric = evaluate_cls(val_result, val_y, metric)
        

        while True:   
            
            # Batch removal is easiest
            indices = list(self.X.index)        
            marked_indices = []

            for index in indices:
                EKNN_indices = list(indices)
                EKNN_indices.remove(index)

                knn_classifier = KNNClassifier(self.k_neighbors)
                knn_classifier.fit(self.X.loc[EKNN_indices], self.y.loc[EKNN_indices])

                # If the prediction wrong, we don't need the point!
                if self.y.loc[index] != knn_classifier.predict(self.X.loc[index]):
                    marked_indices.append(index)

            
            # If no examples have been marked, we break the loop
            
            if not marked_indices:
                if self.verbose:
                    print("No points marked for batch removal. Fit has completed")
                break
            if self.verbose:
                print("Points marked for batch removal at indices: ", marked_indices)
            # Make a new list of indices so we can try out our eval metric
            new_indices = []
            

            for index in indices:
                if index not in marked_indices:
                    new_indices.append(index)

            # If new_indices is empty, then that means every point was misclassified
            
            if not new_indices:
                raise ValueError("All points are misclassified on first run.")
                    
            knn_classifier.fit(self.X.loc[new_indices], self.y.loc[new_indices])
            new_val_result = knn_classifier.predict(val_X)
            new_val_metric = evaluate_cls(new_val_result, val_y, metric)
            
            # If the new indices are not worse, refit the data. Else, break.
            if new_val_metric > val_metric:
                self.X = self.X.loc[new_indices]
                self.y = self.y.loc[new_indices]
                if self.verbose:
                    print(f"Improvement in {metric} detected. \n Old : {val_metric} \n New : {new_val_metric}")
            else:
                if self.verbose:
                    print(f"No improvement in {metric} detected. \n Old : {val_metric} \n New : {new_val_metric}")
                break

        
 
        
    """
    This function predicts points based on the learned dataset. Will default to Euclidean distance (L2 norm) if no features were marked as categorical. If all features are marked as categorical, distance calculations will be a VDM. If some but not all are marked as categorical, a mixed type will be used.
    
    Parameters:
    
    X : Pandas Dataframe of points to be predicted.

    ------
    Returns:
    A pandas Series of predicted points.
   
    
    """
    def predict(self, X):
        if isinstance(X, pd.core.frame.DataFrame):
            for column, dtype in X.dtypes.iteritems():
                if not pd.api.types.is_numeric_dtype(dtype):
                    raise TypeError(f'{column} is not a numeric data type')



            # Get the indices of the k_neighbors for each point in the sample we want to predict
            # We want to keep track of the distance calculations for each point
            k_neighbors_distances = {}


            # This is probably the slowest implementation in history, but here we go:

            predictions = []



            for index, row in X.iterrows():


                distances = self.X.apply(lambda x : np.linalg.norm(row - x), axis = 1)

                # --------------------------------------------
                # Sort by this distance and find out the k closest ones.
                # If k is greater than the actual number of training examples we have, then just use all of them.
                if len(self.X) < self.k_neighbors:
                    neighbors = distances.sort_values()
                else:
                    neighbors = distances.sort_values()[:self.k_neighbors]

                # We get the mode, but sometimes there is more than one mode, so we randomly sample and take whatever we get.
                # If it's only one mode, then we get only one.
                prediction = self.y.loc[neighbors.index].mode().sample().iloc[0]



                predictions.append(prediction)


                # -----------------

            # We can just return the predictions now as a list, or we can add the indices.

            # I think it makes sense to add the indices (?).

            return pd.Series(predictions, index = X.index)

        elif isinstance(X, pd.core.series.Series) :
            
            distances = self.X.apply(lambda x : np.linalg.norm(X - x), axis = 1)

            # --------------------------------------------
            # Sort by this distance and find out the k closest ones.
            # If k is greater than the actual number of training examples we have, then just use all of them.
            if len(self.X) < self.k_neighbors:
                neighbors = distances.sort_values()
            else:
                neighbors = distances.sort_values()[:self.k_neighbors]
            


            prediction = None


            neighbors_output = self.y.loc[neighbors.index]
            
            if self.verbose:
                print("Neighbors for this point are: ", neighbors_output, "with features", self.X.loc[neighbors.index])
            
            prediction = self.y.loc[neighbors.index].mode().sample().iloc[0]

            return prediction
    
            

        
    