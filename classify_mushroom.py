from sklearn import preprocessing
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

class  ClassifyMushroom:
    def __init__ (self, data):
        self.features = data[0]
        self.labels = data[1]

    def encode_features(self):
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
        for col in range(new_features.shape[1]):
            new_features_encoded[:, col] = encoder.fit_transform(new_features[:, col])

        hot_encoder = preprocessing.OneHotEncoder()
        hot_encoder.fit_transform(new_features_encoded)

        random = np.arange(new_features_encoded.shape[0])
        np.random.shuffle(random)
        feature_train = new_features_encoded[random[:len(content)], :]
        feature_test = new_features_encoded[random[len(content):], :]

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