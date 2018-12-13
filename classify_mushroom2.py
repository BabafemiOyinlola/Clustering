from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

class  ClassifyMushroom2:
    def __init__ (self):
        self.features = []
        self.labels = []

    def read_mushroom_data(self, filepath):
        read = open(filepath, "r")
        content = read.readlines()
        if(len(content) == 0):
            return
        else:
            len_content = len(content[0].rstrip().split(",")) #Split the first line in content to obtain number of columns/ features
            data_array = np.empty((len(content), len_content), dtype=str) #Create an empty array to store dataitems
            for line in range(len(content)):
                data_array[line] = content[line].split(",")

            read.close()
            labels = data_array[:, 0]
            features =  np.delete(data_array, obj=0, axis=1)
            # data_array = []
            
            self.features = features
            self.labels = labels
            self.data = data_array

            return (data_array)

    def main_logistic_regression(self):
        x_train, x_test, y_train, y_test = train_test_split(self.features, self.labels, test_size=0.3)

        #train missign data model with training data
        #Fill missing data in training set; x_train
        x_train, x_test = self.predict_missing_data(x_train, x_test)

        #encode features 
        train_features_encoded = np.array([])
        test_features_encoded = np.array([])

        for col in range(x_train.shape[1]):
            temp = pd.get_dummies(x_train[:, col])
            temp = np.array(temp)
            if col == 0:
                train_features_encoded = temp
            else:
                train_features_encoded = np.column_stack([train_features_encoded, temp])

        for col in range(x_test.shape[1]):
            temp = pd.get_dummies(x_test[:, col])
            temp = np.array(temp)
            if col == 0:
                test_features_encoded = temp
            else:
                test_features_encoded = np.column_stack([test_features_encoded, temp])

        #classifier for main model
        classifier = LogisticRegression()
        classifier.fit(train_features_encoded, y_train)  
        pred = classifier.predict(test_features_encoded)

        print("Accuracy:",metrics.accuracy_score(y_test, pred))  

        return
        
    def predict_missing_data(self, train_data, test_data):
        '''In the mushroom dataset, there are missing values. Here, it is handled.'''
        #Select row with missing values and divide into test and training data used to train. 

        get_train_params = self.missing_data(train_data)
        train_features_encoded = get_train_params[0]
        label_train = get_train_params[1]
        train_test_features_encoded = get_train_params[2]

        print("##################### HANDLED MISSING DATA IN TRAIN ####################")
        #predict missing values using random forests
        classifier = RandomForestClassifier(n_estimators=500)
        classifier.fit(train_features_encoded, label_train)
        missing_values = classifier.predict(train_test_features_encoded)

        #fill missing training label with missing values predicted
        count = 0
        for item in range(len(train_data)):
            if train_data[item, 10] == "?":
                train_data[item, 10] = missing_values[count]
                count += 1

        get_test_params = self.missing_data(test_data)
        test_features_encoded = get_test_params[0]
        label_test = get_test_params[1]
        test_features_encoded = get_test_params[2]
        print("Handling missing data in test set")
        missing_values2 = classifier.predict(test_features_encoded)

        print("#####################HANDLED MISSING DATA IN TEST ####################")
        count = 0
        for item in range(len(test_data)):
            if test_data[item, 10] == "?":
                test_data[item, 10] = missing_values2[count]
                count += 1

        return (train_data, test_data)

    def missing_data(self, train_data):
        train_set = []
        test_set = []

        for item in range(len(train_data)):
            if train_data[item, 10] == "?":
                test_set.append(train_data[item])
            else:
                train_set.append(train_data[item])
   
        train_set = np.array(train_set)
        test_set = np.array(test_set)

        #label here is the feature with missing values 
        label_train = train_set[:, 10]
        label_test = test_set[:, 10]

        train_set = np.delete(train_set, obj=10, axis=1)
        test_set = np.delete(test_set, obj=10, axis=1)

        features_combined = np.vstack((train_set, test_set))
        all_feat = np.array([])

        for col in range(features_combined.shape[1]):
            temp = pd.get_dummies(features_combined[:, col])
            temp = np.array(temp)
            if col == 0:
                all_feat = temp
            else:
                all_feat = np.column_stack([all_feat, temp])


        feature_train = all_feat[0:len(train_set)]
        feature_test = all_feat[len(train_set): len(features_combined)]

        train_features_encoded = feature_train
        test_features_encoded = feature_test
    
        return(train_features_encoded, label_train, test_features_encoded)

