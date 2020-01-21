# Applied Machine Learning 3460:489 001
# Spring 2020
# Project 1 - kNN Classifier
# Group: Thomas Li, 

"""
Questions to answer (copied from project page)

1.  How many features/attributes does the dataset have?

2.  What is the class distribution?
    a.  How many instances in class1, how many in class2, ...
    b.  Visual display of the instances (data points) with different classes colored differently
        i.  If you have more than 3 attributes in your dataset, you can plot the data points in reduced dimensions (2 or 3)

3.  Dataset partition (training, testing); how many percent of the data is used for training and how many percent for testing

4.  What distance metric is used?

5.  Testing result; what’s the estimated accuracy of your model on future data
"""

## libraries ##
import pandas as pd

## methods and data structures ##
# calculating distance 
def euclidean_distance(item1, item2):
    return
def manhattan_distance(item1, item2):
    return

# finding neighbors

## program parameters ##
# file properties
dataset_filename = ""
separator = ','

# classification properties


## main program ##
# load dataset
dataset = pd.read_csv(dataset_filename, sep=separator)
dataset.head()

# analyse dataset

# test kNN with various parameters
