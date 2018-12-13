from pre_processing import *
from K_means import *
from classify_mushroom import *
# from classify_mushroom2 import *
from clustering import *
from classification import *

# test = Preprocessing("/Users/oyinlola/Desktop/MSc Data Science/SCC403 - Data Mining/Coursework/Joensuu.txt")
# dataset = test.read_data()
# standardised_data = test.standardize_data(dataset)
# normalised_data = test.normalise_data(standardised_data)
# centralized_data = test.centralise(normalised_data)
# transformed_data = test.PCA(centralized_data)

# kmeans = KMeans(3) # 4 works best for now till iterate is fixed
# clusters = kmeans.assign_cluster(standardised_data)
# clustered_points = kmeans.iterate(standardised_data, 0.0001)
# kmeans.plot(clustered_points, "Clustered data")
# print("end")
# HC = HierachicalClustering(dataset)
# HC.calc()

mushroom =  ClassifyMushroom()
mushroom_data = mushroom.read_mushroom_data("/Users/oyinlola/Desktop/MSc Data Science/SCC403 - Data Mining/Coursework/mushroom.csv")
# mushroom = ClassifyMushroom(mushroom_data)

# mushroom.delete_missing_val_feature()
# mushroom.del_missing_value_rows()
# mushroom.logistic_regression_del_feat()
mushroom.logistic_regression_del_rows()


# mushroom_logistic_reg = mushroom.random_forest_classifier1()
# mushroom_logistic_reg = mushroom.random_forest_classifier2()

# mushroom_logistic_reg = mushroom.logistic_regression2()
# mushroom_logistic_reg = mushroom.logistic_regression1()

# mushroom_logistic_reg = mushroom.logistic_regression2()
# mushroom_logistic_reg = mushroom.logistic_regression2_PCA()

# mushroom = ClassifyMushroom2(mushroom_data)
# mushroom.main_random_forest()