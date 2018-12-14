from clustering import *
from pre_processing import *
from classification import *
from class_imbalance import *
from classify_mushroom import *

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
HC.cluster(standardised_data, 12, "Location dendrogram ")

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
mushroom = ClassifyMushroom()
mushroom_data = mushroom.read_mushroom_data("/Users/oyinlola/Desktop/MSc Data Science/SCC403 - Data Mining/Coursework/mushroom.csv")
mushroom.plot_metrics()
#************************************************************************************************************#
#************************************************************************************************************#


#************************************************************************************************************#
#**************************************************ABALONE***************************************************#
#************************************************************************************************************#
abalone = ClassImbalance()
data = abalone.read_data("/Users/oyinlola/Desktop/MSc Data Science/SCC403 - Data Mining/abalone19.txt")
abalone.plot_imbalance()
abalone.plot_metrics()
#************************************************************************************************************#
#************************************************************************************************************#
