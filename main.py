from pre_processing import *
from K_means import *
from classify_mushroom import *
from clustering import *
from classification import *

test = Preprocessing("/Users/oyinlola/Desktop/MSc Data Science/SCC403 - Data Mining/Coursework/Joensuu.txt")
dataset = test.read_data()
standardised_data = test.standardize_data(dataset)
normalised_data = test.normalise_data(standardised_data)
centralized_data = test.centralise(normalised_data)
transformed_data = test.PCA(centralized_data)


mushroom_init = Preprocessing("/Users/oyinlola/Desktop/MSc Data Science/SCC403 - Data Mining/Coursework/mushroom.csv")
mushroom_data = mushroom_init.read_mushroom_data()
mushroom = ClassifyMushroom(mushroom_data)

mushroom_logistic_reg = mushroom.random_forest_classifier1()
mushroom_logistic_reg = mushroom.random_forest_classifier2()

mushroom_logistic_reg = mushroom.logistic_regression2()
mushroom_logistic_reg = mushroom.logistic_regression1()

# mushroom_logistic_reg = mushroom.logistic_regression2()
# mushroom_logistic_reg = mushroom.logistic_regression2_PCA()




# mush = Classification()
# data = mush.read_data("mushroom.csv")
# mush.split_data()
# mush.random_forest_classifier()

