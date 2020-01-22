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

# header fields
attribute_names = []
class_name = ""

# possible settings
k_vals = (1, 3, 5, 7)
distance_methods = {"Euclidean" : euclidean_distance, 
                    "Manhattan" : manhattan_distance}
test_ratios = (0.1, 0.2, 0.25, 0.33, 0.5, 0.67)


## main program ##
# load dataset
dataset = pd.read_csv(dataset_filename, sep=separator)
dataset.head()

# analyse dataset

# test kNN with various parameters
