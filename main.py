from pre_processing import *
from K_means import *
from classify_mushroom import *
from clustering import *

test = Preprocessing("/Users/oyinlola/Desktop/MSc Data Science/SCC403 - Data Mining/Coursework/Joensuu.txt")
dataset = test.read_data()
standardised_data = test.standardize_data(dataset)
normalised_data = test.normalise_data(standardised_data)
centralized_data = test.centralise(normalised_data)
transformed_data = test.PCA(centralized_data)

# kmeans = KMeans(4) # 4 works best for now till iterate is fixed
# clusters = kmeans.assign_cluster(standardised_data)
# clustered_points = kmeans.iterate(standardised_data, 0.0001)
# kmeans.plot(clustered_points, "Clustered data")

# HC = HierachicalClustering(dataset)
# HC.calc()

mushroom_init = Preprocessing("/Users/oyinlola/Desktop/MSc Data Science/SCC403 - Data Mining/Coursework/mushroom.txt")
mushroom_data = mushroom_init.read_mushroom_data()
mushroom = ClassifyMushroom(mushroom_data)
mushroom_random_forest = mushroom.new_random_forest()
# mushroom_random_forest = mushroom.random_forest_classifier()
# mushroom_logistic_reg = mushroom.logistic_regression()
