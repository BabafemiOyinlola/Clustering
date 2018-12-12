from sklearn import preprocessing
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

class  ClassifyMushroom2:
    def __init__ (self, data):
        self.features = data[0]
        self.labels = data[1]

    def main_random_forest(self):
        x_train, x_test, y_train, y_test = train_test_split(self.features, self.labels, test_size=0.3)

        #train missign data model with training data
        #Fill missing data in training set; x_train
        fill_miss_data = self.handle_missing_data2(x_train, x_test)

        x_train = fill_miss_data[0]
        x_test = fill_miss_data[2]
        missing_data_classifier = fill_miss_data[1]


        ####Using random forests to train a model to predict p and e mushrooms####
        # labels_encoded = y_train
        encode_l = preprocessing.LabelEncoder()
        labels_encoded = encode_l.fit_transform(y_train)

        features_encoded = x_train
        encoder = preprocessing.LabelEncoder()
        for col in range(self.features.shape[1]):
            features_encoded[:, col] = encoder.fit_transform(x_train[:, col])

        hot_encoder = preprocessing.OneHotEncoder()
        features_encoded = hot_encoder.fit_transform(features_encoded)

        classifier = RandomForestClassifier(n_estimators=10)
        classifier.fit(features_encoded, y_train)   #classifier for main model

        #Predict missing values in x_test
        # get_params = self.missing_data(x_test)
        # feature_train = get_params[0]
        # feature_test = get_params[1]
        # content = get_params[2]

        # # missing_data_classifier.fit(feature_train, content)
        # missing_values = missing_data_classifier.predict(feature_test)
        # x_test = self.fill_missing_data(x_test, missing_values)
        
        test_features_encoded = x_test
        for col in range(self.features.shape[1]):
            test_features_encoded[:, col] = encoder.fit_transform(x_test[:, col])

        hot_encoder = preprocessing.OneHotEncoder()
        # features_encoded = np.array(features_encoded)
        test_features_encoded = hot_encoder.fit_transform(features_encoded.toarray())

        pred = classifier.predict(test_features_encoded)

        print("Accuracy:",metrics.accuracy_score(y_test, pred))  

        return
        
    def handle_missing_data2(self, train_data, test_data):
        '''In the mushroom dataset, there are missing values. Here, it is handled.'''
        #Select row with missing values and divide into test and training data used to train. 
        get_params = self.missing_data(train_data)
        feature_train = get_params[0]
        feature_test = get_params[1]
        content = get_params[2]
        print("#####################HANDLED MISSING DATA IN TRAIN####################")
        #predict missing values using random forests
        classifier = RandomForestClassifier(n_estimators=500)
        classifier.fit(feature_train, content)
        missing_values = classifier.predict(feature_test)

        get_params2 = self.missing_data(test_data)
        feature_train2 = get_params2[0]
        feature_test2 = get_params2[1]
        content2 = get_params2[2]
        print("Handling missing data in test set")
        #programme breaks here
        missing_values2 = classifier.predict(feature_test2)

        #Fill features with predicted missing values
        train_data = self.fill_missing_data(train_data, missing_values)
        test_data = self.fill_missing_data(test_data,missing_values2)

        return (train_data, classifier, test_data)

    def missing_data(self, train_data):
        missing_value_col = train_data[:, 10]
        content = []
        val = []

        for item in range(len(missing_value_col)):
            if missing_value_col[item] == "?":
                val.append(missing_value_col[item])
            else:
                content.append(missing_value_col[item])
   
        content = np.array(content)
        val = np.array(val)

        new_features = np.delete(train_data, obj=10, axis=1) #drop col with missing values: '?'

        #encode features 
        new_features_encoded = new_features.copy()

        encoder = preprocessing.LabelEncoder()

        #convert to numerical variables first using label encoder
        for col in range(new_features.shape[1]):
            new_features_encoded[:, col] = encoder.fit_transform(new_features[:, col])

        hot_encoder = preprocessing.OneHotEncoder()
        #now convert using one hot encoder
        new_features_encoded = hot_encoder.fit_transform(new_features_encoded)

        random = np.arange(new_features_encoded.shape[0])
        np.random.shuffle(random)
        feature_train = new_features_encoded[random[:len(content)], :]
        feature_test = new_features_encoded[random[len(content):], :]

        feature_train = feature_train.astype(np.float)
        feature_test = feature_test.astype(np.float)

        return(feature_train, feature_test, content)

    def fill_missing_data(self, train_data, predicted_vals):
        count = 0
        update_features = train_data
        for col in range(train_data.shape[0]):
            if train_data[col, 10] == "?":
                print("Before: ", train_data[col,:])
                update_features[col, 10] = predicted_vals[count]
                count += 1
                print("After: ",update_features[col,:])

        print("Count end: ", count)
        if count == len(predicted_vals):
            train_data  = update_features
            print("Missing values handled.")
        
        return(train_data)


    ############FORMER##############

    def random_forest_classifier(self):
        '''Using random forests'''
        self.handle_missing_data()
        labels_encoded = self.labels.copy()
        encode_l = preprocessing.LabelEncoder()
        labels_encoded = encode_l.fit_transform(self.labels)

        features_encoded = self.features.copy()
        encoder = preprocessing.LabelEncoder()
        for col in range(self.features.shape[1]):
            features_encoded[:, col] = encoder.fit_transform(self.features[:, col])

        hot_encoder = preprocessing.OneHotEncoder()
        hot_encoder.fit_transform(features_encoded)

        x_train, x_test, y_train, y_test = train_test_split(features_encoded, labels_encoded, test_size=0.3)

        classifier = RandomForestClassifier(n_estimators=10)
        classifier.fit(x_train, y_train)
        pred = classifier.predict(x_test)

        print("Accuracy:",metrics.accuracy_score(y_test, pred)) 

        return

    def logistic_regression(self):
        '''Using Logistic Regression'''
        self.handle_missing_data()

        labels_encoded = self.labels.copy()
        encode_l = preprocessing.LabelEncoder()
        labels_encoded = encode_l.fit_transform(self.labels)

        features_encoded = self.features.copy()
        encoder = preprocessing.LabelEncoder()
        for col in range(self.features.shape[1]):
            features_encoded[:, col] = encoder.fit_transform(self.features[:, col])

        hot_encoder = preprocessing.OneHotEncoder()
        hot_encoder.fit_transform(features_encoded)

        features_encoded = features_encoded.astype(np.float)
        labels_encoded = labels_encoded.astype(np.float)

        x_train, x_test, y_train, y_test = train_test_split(features_encoded, labels_encoded, test_size=0.3)

        reg = LogisticRegression()
        reg.fit(x_train, y_train)
        pred = reg.predict(x_test)
        conf_matrix = metrics.confusion_matrix(y_test, pred)
        print("Confusion matrix: ", conf_matrix)
        print("Accuracy:",metrics.accuracy_score(y_test, pred)) 
        return

    def handle_missing_data(self):
        '''In the mushroom dataset, there are missing values. Here, it is handled.'''
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

        #convert to numerical variables first using label encoder
        for col in range(new_features.shape[1]):
            new_features_encoded[:, col] = encoder.fit_transform(new_features[:, col])

        hot_encoder = preprocessing.OneHotEncoder()
        #now convert using one hot encoder
        new_features_encoded = hot_encoder.fit_transform(new_features_encoded)

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
                print("Before: ",self.features[col,:])
                update_features[col, 10] = missing_values[count]
                count += 1
                print("After: ",update_features[col,:])

        print("Count end: ", count)
        if count == len(missing_values):
            self.features = update_features
            print("Missing values handled.")

        #Can't calculate the accuracy of the model since I do not have test values to compare against
        # print("Accuracy:",metrics.accuracy_score(val, missing_values)) 
        return

    def metrics(self):

        return