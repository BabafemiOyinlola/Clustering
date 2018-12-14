import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikitplot as skplt
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn import metrics, preprocessing
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score

class  ClassifyMushroom:
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
        
    #random forest with missing data predicted
    def random_forest_classifier_pred(self):
        '''Using random forests'''
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
        classifier = RandomForestClassifier(n_estimators=500)
        classifier.fit(train_features_encoded, y_train)  
        pred = classifier.predict(test_features_encoded)

        # print("Accuracy:",metrics.accuracy_score(y_test, pred))  

        features_encoded = np.vstack([train_features_encoded, test_features_encoded])
        # labels_encoded = pd.get_dummies(np.vstack([y_train, y_test]))
        labels = np.vstack((y_train[:, None], y_test[:, None]))


        name = "Random forests|Missing value predicted"
        print("Random forests|Missing value predicted|CV")
        # cross_val_acc = self.cross_validation(classifier, features_encoded, labels_encoded)
        cross_val_acc = self.cross_validation(classifier, features_encoded, labels)

        self.metrics(pred, y_test, "e", name)
        return  cross_val_acc, y_test, pred

    #random forest with missing data filled with mode
    def random_forest_classifier_mode(self):
        '''Using random forests'''
        #Fill missing data with mode
        self.fill_missing_data_with_mode()

        # labels_encoded = self.labels.copy()
        # encode_l = preprocessing.LabelEncoder()
        # labels_encoded = encode_l.fit_transform(self.labels)

        features_encoded = self.features.copy()
        encoder = preprocessing.LabelEncoder()
        for col in range(self.features.shape[1]):
            features_encoded[:, col] = encoder.fit_transform(self.features[:, col])

        hot_encoder = preprocessing.OneHotEncoder()
        features_encoded =  hot_encoder.fit_transform(features_encoded)

        x_train, x_test, y_train, y_test = train_test_split(features_encoded, self.labels)

        classifier = RandomForestClassifier(n_estimators=500)
        classifier.fit(x_train, y_train)
        pred = classifier.predict(x_test)

        name = "Random forests|Missing value filled with mode"
        print("Random forests|Missing value filled with mode|CV")

        cross_val_acc = self.cross_validation(classifier, features_encoded, self.labels)
        self.metrics(pred, y_test, "e", name)
        return  cross_val_acc, y_test, pred

   #logistic regression with missing data rows deleted
    def random_forests_del_rows(self):
        '''Using Random forest'''
        self.del_missing_value_rows()

        name = "-Random forest. Missing value rows deleted"

        features_encoded = np.array([])

        for col in range(self.features.shape[1]):
            temp = pd.get_dummies(self.features[:, col])
            temp = np.array(temp)
            if col == 0:
                features_encoded = temp
            else:
                features_encoded = np.column_stack([features_encoded, temp])

        x_train, x_test, y_train, y_test = train_test_split(features_encoded, self.labels)

        classifier = RandomForestClassifier(n_estimators=500)
        classifier.fit(x_train, y_train)
        pred = classifier.predict(x_test)

        print("Random forest|Missing value feature deleted|CV")
        cross_val_acc = self.cross_validation(classifier, features_encoded, self.labels)
        print("Random forest|Missing value feature deleted metrics: ")
        self.metrics(pred, y_test, "e", name)
        return cross_val_acc, y_test, pred

    #logistic regression with missing data feature deleted
    def random_forest_del_feat(self):
        '''Using Random forest'''
        self.delete_missing_val_feature()

        name = "-LRandom forest. Missing featured deleted"

        features_encoded = np.array([])

        for col in range(self.features.shape[1]):
            temp = pd.get_dummies(self.features[:, col])
            temp = np.array(temp)
            if col == 0:
                features_encoded = temp
            else:
                features_encoded = np.column_stack([features_encoded, temp])

        x_train, x_test, y_train, y_test = train_test_split(features_encoded, self.labels)

        classifier = RandomForestClassifier(n_estimators=500)
        classifier.fit(x_train, y_train)
        pred = classifier.predict(x_test)

        print("Random forest|Missing value feature deleted|CV")
        cross_val_acc =  self.cross_validation(classifier, features_encoded, self.labels)
        print("Random forest|Missing value feature deleted metrics: ")
        self.metrics(pred, y_test, "e", name)
        return cross_val_acc, y_test, pred

    #logistic regression with missing data predicted
    def logistic_regression_pred(self):
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
        features_encoded = np.vstack([train_features_encoded, test_features_encoded])
        # labels_encoded = pd.get_dummies(np.concatenate((y_train, y_test), axis=0))

        name = "Logistic regression|Missing value predicted"
        print("Logistic regression|Missing value predicted|CV")

        # features = np.vstack([x_train, x_test])
        labels = np.vstack((y_train[:, None], y_test[:, None]))


        # cross_val_acc = self.cross_validation(classifier, features_encoded, labels_encoded)
        cross_val_acc = self.cross_validation(classifier, features_encoded, labels)

        self.metrics(pred, y_test, "e", name)
        return  cross_val_acc, y_test, pred

    #logistic regression with missing data filled with mode
    def logistic_regression_mode(self):
        '''Using Logistic Regression'''
        self.fill_missing_data_with_mode()

        labels_encoded = self.labels.copy()
        encode_l = preprocessing.LabelEncoder()
        # labels_encoded = encode_l.fit_transform(self.labels)

        features_encoded = self.features.copy()
        encoder = preprocessing.LabelEncoder()
        for col in range(self.features.shape[1]):
            features_encoded[:, col] = encoder.fit_transform(self.features[:, col])

        hot_encoder = preprocessing.OneHotEncoder()
        features_encoded = hot_encoder.fit_transform(features_encoded)

        # features_encoded = features_encoded.astype(np.float)
        # labels_encoded = labels_encoded.astype(np.float)

        x_train, x_test, y_train, y_test = train_test_split(features_encoded, self.labels)

        reg = LogisticRegression()
        reg.fit(x_train, y_train)
        pred = reg.predict(x_test)
         
        print("Logistic Regression|Missing value filled with mode|CV")
        cross_val_acc = self.cross_validation(reg, features_encoded, labels_encoded)
        self.metrics(pred, y_test, "e")
        return cross_val_acc, y_test, pred

    #logistic regression with missing data rows deleted
    def logistic_regression_del_rows(self):
        '''Using Logistic Regression'''
        self.del_missing_value_rows()

        name = "-Logistic Regression. Missing value rows deleted"

        features_encoded = np.array([])

        for col in range(self.features.shape[1]):
            temp = pd.get_dummies(self.features[:, col])
            temp = np.array(temp)
            if col == 0:
                features_encoded = temp
            else:
                features_encoded = np.column_stack([features_encoded, temp])

        x_train, x_test, y_train, y_test = train_test_split(features_encoded, self.labels)

        reg = LogisticRegression()
        reg.fit(x_train, y_train)
        pred = reg.predict(x_test)

        print("Logistic Regression|Missing value feature deleted|CV")
        cross_val_acc = self.cross_validation(reg, features_encoded, self.labels)
        # print("Logistics Regression 1 metrics: ")
        self.metrics(pred, y_test, "e", name)
        return cross_val_acc, y_test, pred

    #logistic regression with missing data feature deleted
    def logistic_regression_del_feat(self):
        '''Using Logistic Regression'''
        self.delete_missing_val_feature()

        # reduce dimentionality
        name = "-Logistic Regression. Missing featured deleted"
        # self.PCA(features_encoded)
        # self.percentage_of_variance(features_encoded, name)

        features_encoded = np.array([])

        for col in range(self.features.shape[1]):
            temp = pd.get_dummies(self.features[:, col])
            temp = np.array(temp)
            if col == 0:
                features_encoded = temp
            else:
                features_encoded = np.column_stack([features_encoded, temp])

        x_train, x_test, y_train, y_test = train_test_split(features_encoded, self.labels)

        reg = LogisticRegression()
        reg.fit(x_train, y_train)
        pred = reg.predict(x_test)

        print("Logistic Regression|Missing value feature deleted|CV")
        cross_val_acc = self.cross_validation(reg, features_encoded, self.labels)
        # print("Logistics Regression 1 metrics: ")
        self.metrics(pred, y_test, "e", name)
        return cross_val_acc, y_test, pred
 

    #random forest with missing data predicted
    def random_forest_classifier_pred_PCA(self):
        '''Using random forests'''
        feat_PCA = self.PCA(self.features)
        x_train, x_test, y_train, y_test = train_test_split(feat_PCA, self.labels, test_size=0.3)

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
        classifier = RandomForestClassifier(n_estimators=500)
        classifier.fit(train_features_encoded, y_train)  
        pred = classifier.predict(test_features_encoded)

        # print("Accuracy:",metrics.accuracy_score(y_test, pred))  

        features_encoded = np.vstack([train_features_encoded, test_features_encoded])
        # labels_encoded = pd.get_dummies(np.vstack([y_train, y_test]))
        labels = np.vstack((y_train[:, None], y_test[:, None]))


        name = "Random forests|Missing value predicted"
        print("Random forests|Missing value predicted|CV")
        # cross_val_acc = self.cross_validation(classifier, features_encoded, labels_encoded)
        cross_val_acc = self.cross_validation(classifier, features_encoded, labels)

        self.metrics(pred, y_test, "e", name)
        return  cross_val_acc, y_test, pred

    #random forest with missing data filled with mode
    def random_forest_classifier_mode_PCA(self):
        '''Using random forests'''
        #Fill missing data with mode
        self.fill_missing_data_with_mode()
        feat_PCA = self.PCA(self.features)

        labels_encoded = self.labels.copy()
        encode_l = preprocessing.LabelEncoder()
        labels_encoded = encode_l.fit_transform(self.labels)

        features_encoded = feat_PCA.copy()
        encoder = preprocessing.LabelEncoder()
        for col in range(feat_PCA.shape[1]):
            features_encoded[:, col] = encoder.fit_transform(feat_PCA[:, col])

        hot_encoder = preprocessing.OneHotEncoder()
        features_encoded =  hot_encoder.fit_transform(features_encoded)

        x_train, x_test, y_train, y_test = train_test_split(features_encoded, labels_encoded)

        classifier = RandomForestClassifier(n_estimators=500)
        classifier.fit(x_train, y_train)
        pred = classifier.predict(x_test)

        name = "Random forests|Missing value filled with mode"
        print("Random forests|Missing value filled with mode|CV")

        cross_val_acc = self.cross_validation(classifier, features_encoded, labels_encoded)
        self.metrics(pred, y_test, 1, name)
        return  cross_val_acc, y_test, pred

   #logistic regression with missing data rows deleted
    def random_forests_del_rows_PCA(self):
        '''Using Random forest'''
        self.del_missing_value_rows()
        feat_PCA = self.PCA(self.features)

        name = "-Random forest. Missing value rows deleted"

        features_encoded = np.array([])

        for col in range(feat_PCA.shape[1]):
            temp = pd.get_dummies(feat_PCA[:, col])
            temp = np.array(temp)
            if col == 0:
                features_encoded = temp
            else:
                features_encoded = np.column_stack([features_encoded, temp])

        x_train, x_test, y_train, y_test = train_test_split(features_encoded, self.labels)

        classifier = RandomForestClassifier(n_estimators=500)
        classifier.fit(x_train, y_train)
        pred = classifier.predict(x_test)

        print("Random forest|Missing value feature deleted|CV")
        cross_val_acc = self.cross_validation(classifier, features_encoded, self.labels)
        print("Random forest|Missing value feature deleted metrics: ")
        self.metrics(pred, y_test, "e", name)
        return cross_val_acc, y_test, pred

    #logistic regression with missing data feature deleted
    def random_forest_del_feat_PCA(self):
        '''Using Random forest'''
        self.delete_missing_val_feature()
        feat_PCA = self.PCA(self.features)

        name = "-Random forest. Missing featured deleted"

        features_encoded = np.array([])

        for col in range(feat_PCA.shape[1]):
            temp = pd.get_dummies(feat_PCA[:, col])
            temp = np.array(temp)
            if col == 0:
                features_encoded = temp
            else:
                features_encoded = np.column_stack([features_encoded, temp])

        x_train, x_test, y_train, y_test = train_test_split(features_encoded, self.labels)

        classifier = RandomForestClassifier(n_estimators=500)
        classifier.fit(x_train, y_train)
        pred = classifier.predict(x_test)

        print("Random forest|Missing value feature deleted|CV")
        cross_val_acc =  self.cross_validation(classifier, features_encoded, self.labels)
        print("Random forest|Missing value feature deleted metrics: ")
        self.metrics(pred, y_test, "e", name)
        return cross_val_acc, y_test, pred

    #logistic regression with missing data predicted
    def logistic_regression_pred_PCA(self):
        feat_PCA = self.PCA(self.features)
        x_train, x_test, y_train, y_test = train_test_split(feat_PCA, self.labels, test_size=0.3)

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
        features_encoded = np.vstack([train_features_encoded, test_features_encoded])
        # labels_encoded = pd.get_dummies(np.concatenate((y_train, y_test), axis=0))

        name = "Logistic regression|Missing value predicted"
        print("Logistic regression|Missing value predicted|CV")

        # features = np.vstack([x_train, x_test])
        labels = np.vstack((y_train[:, None], y_test[:, None]))


        # cross_val_acc = self.cross_validation(classifier, features_encoded, labels_encoded)
        cross_val_acc = self.cross_validation(classifier, features_encoded, labels)

        self.metrics(pred, y_test, "e", name)
        return  cross_val_acc, y_test, pred

    #logistic regression with missing data filled with mode
    def logistic_regression_mode_PCA(self):
        '''Using Logistic Regression'''
        self.fill_missing_data_with_mode()
        feat_PCA = self.PCA(self.features)

        labels_encoded = self.labels.copy()
        encode_l = preprocessing.LabelEncoder()
        labels_encoded = encode_l.fit_transform(self.labels)

        features_encoded = feat_PCA.copy()
        encoder = preprocessing.LabelEncoder()
        for col in range(feat_PCA.shape[1]):
            features_encoded[:, col] = encoder.fit_transform(feat_PCA[:, col])

        hot_encoder = preprocessing.OneHotEncoder()
        features_encoded = hot_encoder.fit_transform(features_encoded)

        features_encoded = features_encoded.astype(np.float)
        labels_encoded = labels_encoded.astype(np.float)

        x_train, x_test, y_train, y_test = train_test_split(features_encoded, labels_encoded)

        reg = LogisticRegression()
        reg.fit(x_train, y_train)
        pred = reg.predict(x_test)
         
        print("Logistic Regression|Missing value filled with mode|CV")
        cross_val_acc = self.cross_validation(reg, features_encoded, labels_encoded)
        self.metrics(pred, y_test, 1)
        return cross_val_acc, y_test, pred

    #logistic regression with missing data rows deleted
    def logistic_regression_del_rows_PCA(self):
        '''Using Logistic Regression'''
        self.del_missing_value_rows()
        feat_PCA = self.PCA(self.features)

        name = "-Logistic Regression. Missing value rows deleted"

        features_encoded = np.array([])

        for col in range(feat_PCA.shape[1]):
            temp = pd.get_dummies(feat_PCA[:, col])
            temp = np.array(temp)
            if col == 0:
                features_encoded = temp
            else:
                features_encoded = np.column_stack([features_encoded, temp])

        x_train, x_test, y_train, y_test = train_test_split(features_encoded, self.labels)

        reg = LogisticRegression()
        reg.fit(x_train, y_train)
        pred = reg.predict(x_test)

        print("Logistic Regression|Missing value feature deleted|CV")
        cross_val_acc = self.cross_validation(reg, features_encoded, self.labels)
        self.metrics(pred, y_test, "e", name)
        return cross_val_acc, y_test, pred

    #logistic regression with missing data feature deleted
    def logistic_regression_del_feat_PCA(self):
        '''Using Logistic Regression'''
        self.delete_missing_val_feature()
        feat_PCA = self.PCA(self.features)

        name = "-Logistic Regression. Missing featured deleted"

        features_encoded = np.array([])

        for col in range(feat_PCA.shape[1]):
            temp = pd.get_dummies(feat_PCA[:, col])
            temp = np.array(temp)
            if col == 0:
                features_encoded = temp
            else:
                features_encoded = np.column_stack([features_encoded, temp])

        x_train, x_test, y_train, y_test = train_test_split(features_encoded, self.labels)

        reg = LogisticRegression()
        reg.fit(x_train, y_train)
        pred = reg.predict(x_test)

        print("Logistic Regression|Missing value feature deleted|CV")
        cross_val_acc = self.cross_validation(reg, features_encoded, self.labels)
        # print("Logistics Regression 1 metrics: ")
        self.metrics(pred, y_test, "e", name)
        return cross_val_acc, y_test, pred
 
    #handle missing data by filling with mode
    def fill_missing_data_with_mode(self):
        miss_col = self.features[:, 10]
        
        feature_dict = {}

        for i in range(miss_col.shape[0]):
            if miss_col[i] in feature_dict:
                feature_dict[miss_col[i]] += 1
            else:
                feature_dict[miss_col[i]] = 1
        
        mode_num = 0
        mode = ""

        for key, value in feature_dict.items():
            if key != "?":
                if value > mode_num:
                    mode_num = value
                    mode = key
        
        #Impute missing values with mode
        count = 0
        for i in range(self.features.shape[0]):
            if self.features[i, 10] == "?":
                self.features[i, 10] = mode
                count += 1

        if mode_num >= count:
            return 
        else:
           print("Missing data not handled")
           return False

    #handle missing data by deleting feature contsining missing values (PCA argument)
    def delete_missing_val_feature(self):
        features = self.features
        new_feautures = np.delete(features, obj=10, axis=1)
        self.features = new_feautures   
        return
    
    def del_missing_value_rows(self):
        data = self.data.copy()
        length = data.shape[0]
        for i in range(length - 1, 0, -1):
            if data[i, 11] == "?":
                data = np.delete(data, obj=i, axis=0)
                length = data.shape[0]
                continue

        labels = data[:, 0]
        features =  np.delete(data, obj=0, axis=1)

        self.features = features
        self.labels = labels

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

        #reset initaial vals
        self.features = np.delete(self.data, obj=0, axis=1)
        self.labels = self.data[:, 0]

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

    def metrics(self, predictions, true_labels, recall_label, name=""):
        accuracy = metrics.accuracy_score(true_labels, predictions)
        confusion_matrix = metrics.confusion_matrix(true_labels, predictions)
        recall = metrics.recall_score(true_labels, predictions, pos_label=recall_label)
        # precision = metrics.pr
        print("Accuracy: ",accuracy)
        print("Confusion matrix: \n", confusion_matrix)
        print("Recall: ", recall)
        # self.roc_curve_acc(true_labels, predictions, name, recall_label)
        return

    def PCA(self, data):
        pca = PCA(n_components=10)
        pca.fit(data)
        cof = pca.components_
        trasform_data = pca.transform(data)
        return trasform_data

    def percentage_of_variance(self, data, name):
        pca = TruncatedSVD(n_components=22) #we have two components
        pca.fit(data)
        colums = ["cap-shape","cap-surface", "cap-color", "bruises?", "odor", "gill-attachment", 
                    "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring",
                    "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", 
                    "veil-color", "ring-number", "ring-type", "spore-print-color", "population", "habitat"]
        plt.bar(colums, pca.explained_variance_ratio_, tick_label= colums)
        plt.xlabel("Principal Component")
        plt.ylabel("% Variance Explained")
        plt.xticks(rotation='vertical')
        plt.tight_layout()
        plt.title("Percentage of Variance")   
        plt.savefig("Percentage of variance " + name + ".jpeg")
        plt.show()
        return

    def cross_validation(self, algorithm, features, labels):
        accuracy = cross_val_score(algorithm, features, labels, scoring='accuracy', cv = 10)
        accuracy_percent = accuracy.mean() * 100
        print("Cross validation accuracy: " , accuracy_percent)
        return accuracy

    def roc_curve_acc(self, Y_test, Y_pred, name, pos_lab):
        false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(Y_test, Y_pred, pos_label=pos_lab)
        roc_auc = metrics.auc(false_positive_rate, true_positive_rate)
        plt.title('ROC Curve')
        plt.plot(false_positive_rate, true_positive_rate, color='darkorange',label='AUC = %0.3f'%(roc_auc))
        plt.legend(loc='lower right')
        plt.plot([0,1],[0,1],'b--')
        plt.ylim([-0.1, 1.1])
        plt.xlim([-0.1, 1.1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.tight_layout()
        plt.title("ROC Curve")   
        plt.savefig("ROC Curve " + name + ".jpeg")
        plt.show()
        return

    def plot_metrics(self):
        # lg_pred = self.logistic_regression_pred()
        lg_mode = self.logistic_regression_mode()
        lg_del_rows = self.logistic_regression_del_rows()
        lg_del_feat = self.logistic_regression_del_feat()


        # dc_pred = self.random_forest_classifier_pred()
        dc_mode = self.random_forest_classifier_mode()
        dc_del_rows = self.logistic_regression_del_rows()
        dc_del_feat = self.logistic_regression_del_feat()


        # lg_mode_PCA = self.logistic_regression_mode_PCA()
        # lg_del_rows_PCA = self.logistic_regression_del_rows_PCA()
        # lg_del_feat_PCA = self.logistic_regression_del_feat_PCA()


        # # dc_pred = self.random_forest_classifier_pred()
        # dc_mode_PCA = self.random_forest_classifier_mode_PCA()
        # dc_del_rows_PCA = self.logistic_regression_del_rows_PCA()
        # dc_del_feat_PCA = self.logistic_regression_del_feat_PCA()


        accuracies = [0,lg_mode[0], lg_del_rows[0], lg_del_feat[0], 0, dc_mode[0], dc_del_rows[0], dc_del_feat[0]]
        accuracies_labels = ["lg_pred","lg_mode", "lg_del_rows", "lg_del_feat","dc_pred", "dc_mode", "dc_del_rows", "dc_del_feat"]

        # accuracies_PCA = [0,lg_mode_PCA[0], lg_del_rows_PCA[0], lg_del_feat_PCA[0], 0, dc_mode_PCA[0], dc_del_rows_PCA[0], dc_del_feat_PCA[0]]
        # accuracies_labels = ["lg_pred_PCA","lg_mode_PCA", "lg_del_rows_PCA", "lg_del_feat_PCA","dc_pred_PCA", "dc_mode_PCA", "dc_del_rows_PCA", "dc_del_feat_PCA"]

        means, means_PCA = [], []
        std_dev, std_dev_PCA = [], []
        x_axis = np.arange(len(accuracies))

        for i in range(len(accuracies)):
            temp = np.array(accuracies[i])
            mean = np.mean(temp)
            std = np.std(temp)
            means.append(mean)
            std_dev.append(std)

        # for i in range(len(accuracies_PCA)):
        #     temp = np.array(accuracies_PCA[i])
        #     mean = np.mean(temp)
        #     std = np.std(temp)
        #     means_PCA.append(mean)
        #     std_dev_PCA.append(std)

       
        width = 0.25
        fig, ax = plt.subplots()
        plt1 = ax.bar(x_axis, means, width,  yerr=std_dev, align='center', alpha=0.5, ecolor='black', capsize=10)
        # plt2 = ax.bar(x_axis+width, means_PCA, width, yerr=std_dev_PCA, align='center', alpha=0.5, ecolor='black', capsize=10, color="darkblue")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xticks(x_axis)
        ax.set_xticklabels(accuracies_labels)
        ax.set_title("Average model accuracy and error")
        ax.yaxis.grid(True)
        # ax.legend((plt1[0], plt2[0]), ("Before PCA", "After PCA"))
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig("Model accuracy for Mushroom Data.png")
        plt.show()

        true_labels = [lg_mode[1], lg_del_rows[1], lg_del_feat[1],dc_mode[1], dc_del_rows[1], dc_del_feat[1]]
        predictions = [lg_mode[2], lg_del_rows[2], lg_del_feat[2],dc_mode[2], dc_del_rows[2], dc_del_feat[2]]
        pos_lbl = [1, "e", "e", 1, "e", "eho"]
        col1 = ["yellow", "m", "grey", "pink", "salmon", "cadetblue"]
        col2 = ["blue", "red", "black", "brown", "green", "cyan"]
        accuracies_labels = ["lg_mode","lg_del_rows", "lg_del_feat", "dc_mode", "dc_del_rows", "dc_del_feat"]

        encoder = LabelEncoder()

        for i in range(len(true_labels) - 1):
            true_labels_new = encoder.fit_transform(true_labels[i])
            predictions_new = encoder.fit_transform(predictions[i])
            false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(true_labels_new,predictions_new, pos_label=1)
            roc_auc = metrics.auc(false_positive_rate, true_positive_rate)
        
            #plt.figure()
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.plot([0, 1], [0, 1], color=col1[i], linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.plot(false_positive_rate, true_positive_rate, color=col2[i], lw=2, label= accuracies_labels[i] + " area = %0.2f)" % roc_auc)
        plt.title("ROC Curve showing various classifiers")
        plt.legend(loc="lower right")
        plt.savefig("ROC Curve multiple curves - Mushroom.jpeg")
        plt.show()
        return

################################################################################
 #handle missing data by predicting values
    def predict_missing_data_former(self):
        '''In the mushroom dataset, there are missing values. Here, it is handled by predicting with
        random forests'''
        #Select row with missing values and divide into test and training data used to train. 
        missing_value_col = self.features[:, 10]
        content = []
        val = []
        for item in range(len(missing_value_col)):
            if missing_value_col[item] == "?":
                val.append(missing_value_col[item])
            else:
                content.append(missing_value_col[item])
   
        content = np.array(content)
        val = np.array(val)

        new_features = np.delete(self.features, obj=10, axis=1) #drop col with missing values: '?'

        #encode features 
        new_features_encoded = new_features.copy()

        encoder = preprocessing.LabelEncoder()
        for col in range(new_features.shape[1]):
            new_features_encoded[:, col] = encoder.fit_transform(new_features[:, col])

        hot_encoder = preprocessing.OneHotEncoder()
        hot_encoder.fit_transform(new_features_encoded)

        random = np.arange(new_features_encoded.shape[0])
        np.random.shuffle(random)
        feature_train = new_features_encoded[random[:len(content)], :]
        feature_test = new_features_encoded[random[len(content):], :]

        feature_train = feature_train.astype(np.float)
        feature_test = feature_test.astype(np.float)

        #predict missing values using random forests
        classifier = RandomForestClassifier(n_estimators=500)
        classifier.fit(feature_train, content)
        missing_values = classifier.predict(feature_test)

        update_features = self.features
        count = 0

        #Fill features with predicted missing values
        for col in range(self.features.shape[0]):
            if self.features[col, 10] == "?":
                # print("Before: ",self.features[col,:])
                update_features[col, 10] = missing_values[count]
                count += 1
                # print("After: ",update_features[col,:])

        # print("Count end: ", count)
        if count == len(missing_values):
            self.features = update_features
            print("Missing values handled.")
 
        return
