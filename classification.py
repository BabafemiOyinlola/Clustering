import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

class Classification:
    def __init__ (self):
        
        return
    
    def read_data(self, filepath):
        colums = ["edibility","cap-shape","cap-surface", "cap-color", "bruises?", "odor", "gill-attachment", 
                    "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring",
                    "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", 
                    "veil-color", "ring-number", "ring-type", "spore-print-color", "population", "habitat"]
        data = pd.read_csv(filepath, names=colums)
        self.data = data

        return data

    def split_data(self):
        label = self.data["edibility"]
        features = self.data.drop(["edibility"], axis=1)
        features_train, features_test, label_train, label_test = train_test_split(features, label, test_size=0.3)
        self.label, self.features = label, features
        self.features_train, self.features_test = features_train, features_test
        self.label_train, self.label_test = label_train, label_test
        return 

    def preprocess_missing_data(self, train_data, data, modelClassifier=None):
        #pre-process missing data in self.features_train       
        new_features_train = train_data.loc[train_data["stalk-root"] != "?"]
        new_features_test = train_data.loc[train_data["stalk-root"] == "?"]
        
        new_label_train = new_features_train[["stalk-root"]]
        new_label_test = new_features_test[["stalk-root"]]
        
        new_features_train = new_features_train.drop(["stalk-root"], axis=1)
        new_features_test = new_features_test.drop(["stalk-root"], axis=1)
        
        len_train_data = len(new_features_train)
        train_and_test = pd.concat(objs=[new_features_train, new_features_test], axis=0)
        train_and_test_encoded = pd.get_dummies(train_and_test)
        
        features_train_preprocessed = train_and_test_encoded[:len_train_data]
        features_test_preprocessed = train_and_test_encoded[len_train_data:]
        
        predictions = []

        if data == "train":
            classifier = RandomForestClassifier(n_estimators=50)
            classifier.fit(features_train_preprocessed, new_label_train)     
            predictions = classifier.predict(features_test_preprocessed)
        
        if data == "test":
            classifier = modelClassifier
            classifier.fit(features_train_preprocessed, new_label_train)     
            predictions = classifier.predict(features_test_preprocessed)

        #fill missing values in the test data and insert to the right column
        new_features_test.insert(loc=10, column='stalk-root', value=predictions)
        #fill the train feature set with the train label set
        new_features_train.insert(loc=10, column='stalk-root', value=new_label_train)

        #combine all features
        all_features = pd.concat([new_features_train, new_features_test])         

        return (all_features, classifier)


    def random_forest_classifier2(self):
        train_data = self.features_train
        test_data = self.features_test

        misd_data_model = self.preprocess_missing_data(train_data, "train")

        train_data = misd_data_model[0]
        misd_classifier = misd_data_model[1]


        len_train_data = len(train_data)
        train_and_test = pd.concat(objs=[train_data, test_data], axis=0)
        train_and_test_encoded = pd.get_dummies(train_and_test)
        
        features_train_preprocessed = train_and_test_encoded[:len_train_data]

        classifier = RandomForestClassifier(n_estimators=50)
        classifier.fit(features_train_preprocessed, self.label_train)

        test_data = self.preprocess_missing_data(test_data, "test", misd_classifier)
        test_data = test_data[0]

        len_test_data = len(test_data)
        train_and_test = pd.concat(objs=[test_data, train_data], axis=0)
        train_and_test_encoded = pd.get_dummies(train_and_test)

        features_test_preprocessed = train_and_test_encoded[:len_test_data]
 
        predictions = classifier.predict(features_test_preprocessed)

        print("Accuracy:",metrics.accuracy_score(self.label_test, predictions)) 

        return


