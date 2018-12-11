from pre_processing import *
# from K_means import *
# from classify_mushroom import *
from clustering import *
from classification import *
from classify_mushroom2 import *
# from kmeans import *

test = Preprocessing("/Users/oyinlola/Desktop/MSc Data Science/SCC403 - Data Mining/Coursework/Joensuu.txt")
dataset = test.read_data()
standardised_data = test.standardize_data(dataset)
normalised_data = test.normalise_data(standardised_data)
centralized_data = test.centralise(normalised_data)
transformed_data = test.PCA(centralized_data)


kmeans = KMeans(6)
kmeans.iterate(standardised_data)
kmeans.plot("Clustered Data")




# kmeans = KMeans(3) # 4 works best for now till iterate is fixed
# clusters = kmeans.assign_cluster(standardised_data)
# clustered_points = kmeans.iterate(standardised_data, 0.0001)
# kmeans.plot(clustered_points, "Clustered data")
# print("end")
# HC = HierachicalClustering(dataset)
# HC.calc()

mushroom_init = Preprocessing("/Users/oyinlola/Desktop/MSc Data Science/SCC403 - Data Mining/Coursework/mushroom.csv")
mushroom_data = mushroom_init.read_mushroom_data()
# mushroom = ClassifyMushroom(mushroom_data)
mushroom = ClassifyMushroom2(mushroom_data)

# mushroom_logistic_reg = mushroom.random_forest_classifier1()
# mushroom_logistic_reg = mushroom.random_forest_classifier2()

# mushroom_logistic_reg = mushroom.logistic_regression2()
# mushroom_logistic_reg = mushroom.logistic_regression1()

# mushroom_logistic_reg = mushroom.logistic_regression2()
# mushroom_logistic_reg = mushroom.logistic_regression2_PCA()


# mush0 = mushroom.new_random_forest()
# mush1 = mushroom.random_forest_classifier()
# mush2 = mushroom.logistic_regression()



# mush = Classification()
# data = mush.read_data("mushroom.csv")
# mush.split_data()
# mush.random_forest_classifier()

