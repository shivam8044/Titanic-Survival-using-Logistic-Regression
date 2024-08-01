# Titanic-Survival-using-Logistic-Regression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score,recall_score,mean_squared_error,accuracy_score
from sklearn.model_selection import train_test_split
dataset = pd.read_csv('Titanic-Dataset.csv')
dataset.shape
dataset.info()
dataset.drop(columns=["Cabin"],inplace=True)
dataset.fillna(dataset['Age'].mean(),inplace=True)
dataset.info()
dataset.shape
dataset.columns
dataset.info()
dataset['Fare']=dataset['Fare'].astype('int64')
dataset['Age']=dataset['Age'].astype('int64')
dataset.info()
numerical_features = dataset.columns[dataset.dtypes == 'int64']
numerical_features
numerical_features_df = dataset[numerical_features]
numerical_features_df
plt.scatter(numerical_features_df['Survived'],numerical_features_df['Fare'],
           ls=':'
            ,marker='8'
            ,color='red'
            ,alpha=0.2
            ,linewidths=0.1)
np.corrcoef(numerical_features_df['Survived'],numerical_features_df['Fare'])[0][1]
features_list = []

def get_corr(a='Survived'):
    
    for i in numerical_features_df.columns:
        corr_val = np.corrcoef(numerical_features_df[a],numerical_features_df[i])[0][1]
        if corr_val > -0.1:
            features_list.append(i)
get_corr()
features_list
best_features = numerical_features_df[features_list]
best_features_predictors = best_features[best_features.columns[best_features.columns != 'Survived']]
best_features_predictors
best_features_target# split the data

features_train ,features_test ,targets_train ,targets_test = train_test_split(best_features_predictors,
                                                                         best_features_target,
                                                                        test_size=0.3,
                                                                        shuffle=True,
                                                                        random_state=42)
  model = LogisticRegression(max_iter=100)
  model.fit(features_train,targets_train)
  target_pred = model.predict(features_test)
  target_pred
  targets_test['predicted_target'] = target_pred
  accuracy = sum(targets_test['Survived'] == targets_test['predicted_target'])/targets_test.shape[0] * 100
  accuracy
  recall_score(targets_test['Survived'],targets_test['predicted_target'])
  precision_score(targets_test['Survived'],targets_test['predicted_target'])
  sklearn.metrics.confusion_matrix(targets_test['Survived'],targets_test['predicted_target'])
  plt.hist(targets_test['Survived'])
plt.xlabel('Survived vs not-survived')
plt.ylabel('frequency of passengers survived vs not-survived')
plt.title('True Values histogram');
plt.hist(targets_test['predicted_target'])
plt.xlabel('Survived vs not-survived')
plt.ylabel('frequency of passengers survived vs not-survived')
plt.title('Predicted Values histogram');
