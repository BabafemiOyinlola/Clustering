from clustering import *
# from classify_mushroom2 import *
from pre_processing import *
from classify_mushroom import *
from class_imbalance import *
from classification import *

test = Preprocessing("/Users/oyinlola/Desktop/MSc Data Science/SCC403 - Data Mining/Coursework/Joensuu.txt")
dataset = test.read_data_Joensuu()
standardised_data = test.standardize_data(dataset)
normalised_data = test.normalise_data(standardised_data)
centralized_data = test.centralise(normalised_data)
transformed_data = test.PCA(centralized_data)

#Save plots for data and standardized data
test.plot(dataset, "Latitude", "Longitude", "User Location - JOENSUU")
test.plot(standardised_data, "Latitude", "Longitude", "Standardised Location Data")

# kmeans = KMeans(4)
# kmeans.iterate(standardised_data)
# kmeans.plot("K-means Clustered Data")

#Elbow method
elbow_method = ElbowMethod()
k = range(1, 10)
errors = []

for i in k:
    kmeans = KMeans(i)
    kmeans.iterate(standardised_data)

    sse = elbow_method.sum_of_squared_errors(kmeans.grouped_points, kmeans.centroids)
    errors.append(sse)

plt.plot(k, errors, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

#************************************************************************************************************#
#**********************************************CLUSTERING****************************************************#
#************************************************************************************************************#
#K-means clustering
kmeans = KMeans(4)
kmeans.iterate(standardised_data)
kmeans.plot("K-means Clustered Data")

#Hierachical clustering
HC = HierachicalClustering(standardised_data)
HC.cluster(standardised_data)
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

