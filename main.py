from clustering import *
from classify_mushroom2 import *
from classify_mushroom import *
# from class_imbalance import *
from pre_processing import *

test = Preprocessing("/Users/oyinlola/Desktop/MSc Data Science/SCC403 - Data Mining/Coursework/Joensuu.txt")
dataset = test.read_data_Joensuu()
standardised_data = test.standardize_data(dataset)
no_outliers_data = test.remove_outliers(standardised_data)
normalised_data = test.normalise_data(standardised_data)
centralized_data = test.centralise(normalised_data)
transformed_data = test.PCA(centralized_data)

#Save plots for data and standardized data
test.plot(dataset, "Latitude", "Longitude", "User Location - JOENSUU")
test.plot(standardised_data, "Latitude", "Longitude", "Standardised Location Data")

#************************************************************************************************************#
#**********************************************CLUSTERING****************************************************#
#************************************************************************************************************#
#K-means clustering
kmeans = KMeans(4)
kmeans.iterate(standardised_data)
kmeans.plot("K-means Clustered Data")

#Hierachical clustering
HC = HierachicalClustering(standardised_data)
HC.cluster(standardised_data, 12, "Location dendrogram")

#K-means clustering without outliers
kmeans = KMeans(4)
kmeans.iterate(no_outliers_data)
kmeans.plot("K-means Clustered Data - Without Outliers")

#Hierachical clustering without outliers
HC = HierachicalClustering(no_outliers_data)
HC.cluster(no_outliers_data, 12, "Location dendrogram without outliers ")
#************************************************************************************************************#
#************************************************************************************************************#



#************************************************************************************************************#
#**************************************************MUSHROOM**************************************************#
#************************************************************************************************************#
mushroom_init = Preprocessing("/Users/oyinlola/Desktop/MSc Data Science/SCC403 - Data Mining/Coursework/mushroom.txt")
mushroom_data = mushroom_init.read_mushroom_data()
mushroom = ClassifyMushroom(mushroom_data)
# pred1 = mushroom.random_forest_classifier()
pred2 = mushroom.logistic_regression()
#************************************************************************************************************#
#************************************************************************************************************#


#************************************************************************************************************#
#**************************************************ABALONE***************************************************#
#************************************************************************************************************#
abalone = ClassImbalance()
abalone.read_data("/Users/oyinlola/Desktop/MSc Data Science/SCC403 - Data Mining/Coursework/abalone19.csv")

abalone.logistic_regression_oversampled()
abalone.logistic_regression_oversampled_PCA()
abalone.logistic_regression_undersampled()
abalone.logistic_regression_undersampled_PCA()
abalone.decision_tree_oversampled()
abalone.decision_tree_oversampled_PCA()
abalone.decision_tree_undersampled()
abalone.decision_tree_undersampled_PCA()
#************************************************************************************************************#
#************************************************************************************************************#