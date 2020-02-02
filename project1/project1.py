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
# import matplotlib
import matplotlib.pyplot as plt
# preprocessing
# from sklearn.preprocessing import RobustScaler
# running model evaluation
from sklearn.model_selection import train_test_split
# running kNN
from sklearn.neighbors import KNeighborsClassifier
# getting confusion matrix and accuracy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

## methods and data structures ##
"""
# calculating distance 
def euclidean_distance(item1, item2):
    return
def manhattan_distance(item1, item2):
    return

# finding neighbors
"""
# visualizing results

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
# comment attributes out to ignore them
attribute_names = [#'Time',
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
 #'Amount'
 ]
class_name = "Class"

class_value_labels = {0:"Genuine", 1:"Fraudulent"}
# """

# seed for selecting testing dataset
# and for undersampling dataset
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

# scale attributes (not in use)
"""
scaler = RobustScaler()
dataset["Time"] = scaler.fit_transform(dataset["Time"].values.reshape(-1, 1))
dataset["Amount"] = scaler.fit_transform(dataset["Amount"].values.reshape(-1, 1))
"""

class_members = {c:dataset.loc[dataset[class_name] == c] for c in class_value_labels}

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
        
# give option to undersample dataset
# """
if input("Undersample dataset to balance classes? (y/n) ") in ["y", "Y"]:
    class_size = min([len(class_members[c]) for c in class_members])
    dataset = pd.DataFrame()
    for c in class_members:
        class_members[c] = class_members[c].sample(class_size, random_state=random_seed)
        dataset = dataset.append(class_members[c])
    print("Dataset undersampled to size {0}".format(len(dataset)))
    pass
# """
    
# split into input and target attributes
x = dataset[attribute_names]
y = dataset[class_name]

# display visualizations of dataset properties
# """
# if input("\nShow visualizations for data distributions? (y/n) ") in ["Y", "y"]:
# print("Generating ...")
    
input("Press ENTER to continue to Data Visualizations")

for i in range (0, len(attribute_names), 2):
    # initialize plots, handle sizing and spacing
    figure, axes = plt.subplots(nrows=1, ncols=2)
    figure.tight_layout(pad=3.5)
    
    for j in range(0, 2):
        # handle subplot indexing
        i = i + j
        if i == len(attribute_names):
            break

        # create boxplot
        subplot = axes[j]
        subplot.set_title("{0} vs Class".format(attribute_names[i]))
        subplot.set_ylabel("{0} Value".format(attribute_names[i]))
        subplot.set_xlabel("Class")
        subplot.boxplot([class_members[c][attribute_names[i]] for c in class_members], labels=[class_value_labels[c] for c in class_value_labels], whis=[5, 95], widths=0.5)

    # display plots
    plt.show()

# """
    
input("\nPress [ENTER] to continue to Evaluation")

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
            score = accuracy_score(y_test, y_predicted) 
            
            # normalize confusion matrix to give classes equal weight
            balanced_matrix = confusion_matrix(y_test, y_predicted, normalize='true')
            balanced_score = (balanced_matrix[0, 0] + balanced_matrix[1, 1]) / np.sum(balanced_matrix)
            
            # display and record results
            print("\nkNN with test ratio {0}, k = {1}, {2} distance metric".format(r, k, m))
            print("\nConfusion Matrix (Unnormalized):")
            print(matrix)
            print("Score (Unnormalized): {0}".format(score))
            print("\nConfusion Matrix (Normalized):")
            print(balanced_matrix)
            print("Score (Normalized): {0}\n".format(balanced_score))
            
            
            results.append({"ratio": r, "k": k, "metric": m, 
                            "score": score, "matrix": matrix,
                            "balanced score": balanced_score,
                            "balanced matrix": balanced_matrix})
            
    pass

# analyse five best-performing variations
# this is a blatant violation of "Don't Repeat Yourself" won't work for other datasets but I didn't feel like figuring out how to generalize this
# """
input("Press ENTER to continue to Analysis")

results.sort(key=lambda x:x["score"])
print("\n\nTop Five Variations (Unweighted Accuracy)")
for result in results[-5:]:
    print("\nTest ratio {0}, k = {1}, {2} distance metric".format(result["ratio"], result["k"], result["metric"]))
    print("Score: {0}".format(result["score"]))
    print("{0}% of genuine transactions predicted".format(100 * result["matrix"][0][0] / (result["matrix"][0][0] + result["matrix"][0][1])))
    print("{0}% of fraudulent transactions predicted".format(100 * result["matrix"][1][1] / (result["matrix"][1][0] + result["matrix"][1][1])))
    
    
print("\n\nFive Lowest-Scoring Variations (Unweighted Accuracy)")
for result in results[:5]:
    print("\nTest ratio {0}, k = {1}, {2} distance metric".format(result["ratio"], result["k"], result["metric"]))
    print("Score: {0}".format(result["score"]))
    print("{0}% of genuine transactions predicted".format(100 * result["matrix"][0][0] / (result["matrix"][0][0] + result["matrix"][0][1])))
    print("{0}% of fraudulent transactions predicted".format(100 * result["matrix"][1][1] / (result["matrix"][1][0] + result["matrix"][1][1])))
    
# """

results.sort(key=lambda x:x["balanced score"])
print("\n\nTop Five Variations (Weighted Accuracy)")
for result in results[-5:]:
    print("\nTest ratio {0}, k = {1}, {2} distance metric".format(result["ratio"], result["k"], result["metric"]))
    print("Score: {0}".format(result["balanced score"]))
    print("{0}% of genuine transactions predicted".format(100 * result["balanced matrix"][0][0] / (result["balanced matrix"][0][0] + result["balanced matrix"][0][1])))
    print("{0}% of fraudulent transactions predicted".format(100 * result["balanced matrix"][1][1] / (result["balanced matrix"][1][0] + result["balanced matrix"][1][1])))
    
print("\n\nFive Lowest-Scoring Variations (Weighted Accuracy)")
for result in results[:5]:
    print("\nTest ratio {0}, k = {1}, {2} distance metric".format(result["ratio"], result["k"], result["metric"]))
    print("Score: {0}".format(result["balanced score"]))
    print("{0}% of genuine transactions predicted".format(100 * result["balanced matrix"][0][0] / (result["balanced matrix"][0][0] + result["balanced matrix"][0][1])))
    print("{0}% of fraudulent transactions predicted".format(100 * result["balanced matrix"][1][1] / (result["balanced matrix"][1][0] + result["balanced matrix"][1][1])))
    
# """
    
# visualize results

# k-value and distance metric vs performance
# initialize plot
figure, axes = plt.subplots(ncols=2)
figure.tight_layout(pad=3.5)
subplot = axes[0]
subplot.set_title("k-value vs Performance")
subplot.set_ylabel("Weighted Accuracy")
subplot.set_xlabel("k")
subplot.set_xticks(k_vals)
for metric in distance_metrics:
    subplot.scatter([r["k"] for r in results if r["metric"] == metric], [r["balanced score"] for r in results if r["metric"] == metric], label=metric, alpha=0.6)
subplot.legend()


# test ratio and distance metric vs performance
# initialize plot
subplot = axes[1]
subplot.set_title("Test Ratio vs Performance")
subplot.set_ylabel("Weighted Accuracy")
subplot.set_xlabel("Test Ratio")
subplot.set_xticks(test_ratios)
for metric in distance_metrics:
    subplot.scatter([r["ratio"] for r in results if r["metric"] == metric], [r["balanced score"] for r in results if r["metric"] == metric], label=metric, alpha=0.6)
subplot.legend()

plt.show()

# test ratio, k-value, and distance metric vs performance (3d plot)
from mpl_toolkits.mplot3d import axes3d
# initialize plot
figure = plt.figure()
subplot = figure.add_subplot(111, projection='3d')
subplot.set_title("Test Ratio and k-value vs Performance")
subplot.set_zlabel("Weighted Accuracy")
subplot.set_ylabel("Test Ratio")
subplot.set_yticks(test_ratios)
subplot.set_xlabel("k")
subplot.set_xticks(k_vals)
for metric in distance_metrics:
    subplot.scatter([r["k"] for r in results if r["metric"] == metric], [r["ratio"] for r in results if r["metric"] == metric], [r["balanced score"] for r in results if r["metric"] == metric], label=metric, alpha=0.6)
subplot.legend()
plt.show()

