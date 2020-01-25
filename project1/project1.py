# Applied Machine Learning 3460:489 001
# Spring 2020
# Project 1 - kNN Classifier
# Group: Thomas Li, 

# This program contains the main code for this project, including 
# that of loading the dataset, analyzing the statistical properties, 
# and running and evaluating variations of kNN

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
# getting confusion matrix
from sklearn.metrics import confusion_matrix


## methods and data structures ##
"""
# calculating distance 
def euclidean_distance(item1, item2):
    return
def manhattan_distance(item1, item2):
    return

# finding neighbors
"""
# calculating accuracy from confusion matrix
def get_matrix_accuracy(matrix):
    return (matrix[0, 0] + matrix[1, 1]) / float(np.sum(matrix))


## program parameters ##
# file properties
"""
# For Wisconsin Breast Cancer dataset:
# (found at https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)
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
# For Credit Card Fraud Dataset:
# (found at https://www.kaggle.com/mlg-ulb/creditcardfraud)
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
k_vals = tuple([i for i in range(0, 10) if i % 2 == 1])
"""
distance_methods = {"Euclidean" : euclidean_distance, 
                    "Manhattan" : manhattan_distance}
"""
# distance metric labels used in sklearn kNN
distance_metrics = ["euclidean", "manhattan", "chebyshev"]
test_ratios = (0.1, 0.2, 0.3, 0.4, 0.5)


## main program ##
# load dataset

dataset = pd.read_csv(dataset_filename, sep=separator)
dataset.head()

x = dataset[attribute_names]
y = dataset[class_name]

class_members = {c:[val for val in dataset[class_name] if val == c] for c in class_value_labels}

# analyse dataset
print("Dataset loaded from file {0}".format(dataset_filename))

print("\nAttribute list: {0}".format(attribute_names))

print("\nClass list: {0}".format([class_value_labels[c] for c in class_value_labels]))

print("\n{0} total numeric attributes".format(len(attribute_names)))
print("{0} total class labels".format(len(class_value_labels)))

print("\nItems in dataset: {0}".format(len(dataset)))

print("\nDistribution by class: ")
for c in class_value_labels:
    print("{0}: {1}".format(class_value_labels[c], len(class_members[c])))


# kNN with various parameters
# record results for each combination
results = []
# test different train-test splits
for r in test_ratios:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = r, random_state = random_seed)
    # test different distance metrics
    for m in distance_metrics:
        # test different values of k
        for k in k_vals:
            # set up classifier
            kNN = KNeighborsClassifier(n_neighbors=k, metric=m)
            kNN.fit(x_train, y_train)
            
            # get confusion matrix
            y_predicted = kNN.predict(x_test)
            matrix = confusion_matrix(y_test, y_predicted)
            
            # display and record results
            print("\nkNN with test ratio {0}, k = {1}, {2} distance metric".format(r, k, m))
            results.append({"ratio": r, "k": k, "metric": m})
            
            # display score without balancing
            print("Confusion Matrix (Imbalanced):")
            print(matrix)
            score = get_matrix_accuracy(matrix)
            print("Score (Imbalanced): {0}".format(score))
            results[-1]["score"] = score
            results[-1]["matrix"] = matrix
            
            # balance confusion matrix classes and display again
            print("Confusion Matrix (Balanced):")
            ratio = np.sum(matrix[0]) / np.sum(matrix[1])
            matrix[1, 0] *= ratio
            matrix[1, 1] *= ratio
            print(matrix)
            score = get_matrix_accuracy(matrix)
            print("Score (Balanced): {0}".format(score))
            results[-1]["balanced score"] = score
            results[-1]["balanced matrix"] = matrix
            
    pass

# visualize results
cmap = matplotlib.cm.get_cmap('gnuplot')

# analyse five best-performing variations
results.sort(key=lambda x:x["score"])
top_five = results[-5:]
results.sort(key=lambda x:x["balanced score"])
top_five_balanced = results[-5:]
