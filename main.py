from pre_processing import *
from clustering import *
from classify_mushroom2 import *
from K_means import *
from classify_mushroom import *

test = Preprocessing("/Users/oyinlola/Desktop/MSc Data Science/SCC403 - Data Mining/Coursework/Joensuu.txt")
dataset = test.read_data_Joensuu()
standardised_data = test.standardize_data(dataset)
normalised_data = test.normalise_data(standardised_data)
centralized_data = test.centralise(normalised_data)
transformed_data = test.PCA(centralized_data)

#Save plots for data and standardized data
test.plot(dataset, "Latitude", "Longitude", "User Location - JOENSUU")
test.plot(standardised_data, "Latitude", "Longitude", "Standardised Location Data")

kmeans = KMeans(4)
kmeans.iterate(standardised_data)
kmeans.plot("K-means Clustered Data")

HC = HierachicalClustering(standardised_data)
HC.cluster(standardised_data)

