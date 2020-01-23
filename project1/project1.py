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
# data handling and analysis
import pandas as pd
# statistics
import numpy as np
# visualizations
import matplotlib
# running model evaluation
from sklearn.model_selection import train_test_split
# running kNN
from sklearn.neighbors import KNeighborsClassifier


## methods and data structures ##
# calculating distance 
def euclidean_distance(item1, item2):
    return
def manhattan_distance(item1, item2):
    return

# finding neighbors


## program parameters ##
# file properties
"""
# For Wisconsin Breast Cancer dataset: 
dataset_filename = "wi_breast_cancer.csv"
separator = ','
# header fields
attribute_names = ['radius_mean',
 'texture_mean',
 'perimeter_mean',
 'area_mean',
 'smoothness_mean',
 'compactness_mean',
 'concavity_mean',
 'concave points_mean',
 'symmetry_mean',
 'fractal_dimension_mean',
 'radius_se',
 'texture_se',
 'perimeter_se',
 'area_se',
 'smoothness_se',
 'compactness_se',
 'concavity_se',
 'concave points_se',
 'symmetry_se',
 'fractal_dimension_se',
 'radius_worst',
 'texture_worst',
 'perimeter_worst',
 'area_worst',
 'smoothness_worst',
 'compactness_worst',
 'concavity_worst',
 'concave points_worst',
 'symmetry_worst',
 'fractal_dimension_worst']
class_name = "diagnosis"

class_value_labels = {"M":"Malignant", "B":"Benign"}
"""

# """
# For Credit Card Fraud Dataset
dataset_filename = "creditcard.csv"
separator = ','
# header fields
attribute_names = ['Time',
 'V1',
 'V2',
 'V3',
 'V4',
 'V5',
 'V6',
 'V7',
 'V8',
 'V9',
 'V10',
 'V11',
 'V12',
 'V13',
 'V14',
 'V15',
 'V16',
 'V17',
 'V18',
 'V19',
 'V20',
 'V21',
 'V22',
 'V23',
 'V24',
 'V25',
 'V26',
 'V27',
 'V28',
 'Amount']
class_name = "Class"

class_value_labels = {0:"Genuine", 1:"Fraudulent"}
# """

# seed for selecting testing dataset
random_seed = 42

# possible settings
k_vals = tuple([i for i in range(0, 20) if i % 2 == 0])
distance_methods = {"Euclidean" : euclidean_distance, 
                    "Manhattan" : manhattan_distance}
test_ratios = (0.1, 0.2, 0.25, 0.33, 0.5, 0.67)


## main program ##
# load dataset

dataset = pd.read_csv(dataset_filename, sep=separator)
dataset.head()

x = dataset[attribute_names]
y = dataset[class_name]

# analyse dataset

# test kNN with various parameters
# train-test splits
for r in test_ratios:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = r, random_state = random_seed)
    pass