from pre_processing import *
<<<<<<< HEAD
from clustering import *
from classify_mushroom2 import *
=======
from K_means import *
from classify_mushroom import *
>>>>>>> c13075c25b6c397735869dca1ec317d8ffe2ac3f

test = Preprocessing("/Users/oyinlola/Desktop/MSc Data Science/SCC403 - Data Mining/Coursework/Joensuu.txt")
dataset = test.read_data_Joensuu()
standardised_data = test.standardize_data(dataset)
normalised_data = test.normalise_data(standardised_data)
centralized_data = test.centralise(normalised_data)
transformed_data = test.PCA(centralized_data)

<<<<<<< HEAD
#Save plots for data and standardized data
test.plot(dataset, "Latitude", "Longitude", "User Location - JOENSUU")
test.plot(standardised_data, "Latitude", "Longitude", "Standardised Location Data")
=======
# kmeans = KMeans(4) # 4 works best for now till iterate is fixed
# clusters = kmeans.assign_cluster(standardised_data)
# clustered_points = kmeans.iterate(standardised_data, 0.0001)
# kmeans.plot(clustered_points, "Clustered data")
>>>>>>> c13075c25b6c397735869dca1ec317d8ffe2ac3f

kmeans = KMeans(4)
kmeans.iterate(standardised_data)
kmeans.plot("K-means Clustered Data")

<<<<<<< HEAD
HC = HierachicalClustering(standardised_data)
HC.cluster(standardised_data)
=======
mushroom_init = Preprocessing("/Users/oyinlola/Desktop/MSc Data Science/SCC403 - Data Mining/Coursework/mushroom.txt")
mushroom_data = mushroom_init.read_mushroom_data()
mushroom = ClassifyMushroom(mushroom_data)
# pred1 = mushroom.random_forest_classifier()
pred2 = mushroom.logistic_regression()
>>>>>>> c13075c25b6c397735869dca1ec317d8ffe2ac3f
