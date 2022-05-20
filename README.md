# neural-networks


Here I've implemented a (somewhat inefficient) multilayer feedforward network from scratch for both classification and regression, along with an auto-encoder to compare performance between different model types.

Five fold cross validation is utilized to configure hyperparameters and evaluate model performance. For the classification tasks, model performance is evaluated through a simple accuracy measure, and for regression tasks, model performance is evaluated through mean squared error. Performance between models is compared through a statistical *t*-test for comparison of two means. 

All models are also compared against a "baseline", simple majority model. For regression tasks, this means that we use the mean response across all training data. For classification tasks, it simply outputs the most common class. 

Full writeup and discussion of results is in the [Neural_Networks_Writeup.pdf](/Neural_Networks_Writeup.pdf) file.

The datasets used are in the datasets folder. These are public datasets available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php).

One ipynb is provided as to how the results were calculated (through various Jupyter Notebooks), which is of the Abalone dataset.
