import os,sys
import pandas as pd
import numpy as np
from sklearn import linear_model
import sklearn
import computing_loss

## loading data
### loading train and validataion data
print('loading data')
train_data = pd.read_csv('../data/new_train.csv')
val_data = pd.read_csv('../data/validation.csv')
test_data = pd.read_csv('../data/test.csv')
test_id = test_data['ID']

train_label = train_data['target']
train_data = train_data.loc[:, 'v1':'v131']

val_label = val_data['target']
val_data = val_data.loc[:, 'v1':'v131']

def categary_to_number():
    for column_name in train_data.columns:
        if train_data.dtypes[column_name] == 'object':
            all_possible_value = []
            train_possible_value = np.unique(train_data[column_name])
            test_possible_value = np.unique(test_data[column_name])
            val_possible_value = np.unique(val_data[column_name])
            all_possible_value = np.concatenate((train_possible_value, test_possible_value, val_possible_value))
            all_possible_value = np.unique(all_possible_value)
            m = {}
            for i in range(t.size):
                m[t[i]] = i
            train_data[column_name] = train_data[column_name].map(m)
            test_data[column_name] = test_data[column_name].map(m)
            val_data[column_name] = val_data[column_name].map(m)


def drop_non_numberic(train_data, other_data):
    numberic_columns = []
    for column_name in train_data.columns:
        if train_data.dtypes[column_name] == 'object':
            continue
        numberic_columns.append(column_name)
    
    res_data = other_data.loc[:, numberic_columns]
    return res_data

def fill_nan(train_data, other_data):
    for col in other_data.columns:
        mean_val = train_data[col].mean()
        other_data[col] = other_data[col].fillna(mean_val)
    return other_data


## drop the column whose data type is not numberic
print('drop non-numberic columns')
train_data = drop_non_numberic(train_data, train_data)
val_data = drop_non_numberic(train_data, val_data)
test_data = drop_non_numberic(train_data, test_data)

## using mean value to replace nan
train_data = fill_nan(train_data, train_data)
val_data = fill_nan(train_data, val_data)
test_data = fill_nan(train_data, test_data)

print('training logistic regression model')
logistic = sklearn.linear_model.LogisticRegression(verbose = 1)
logistic.fit(train_data, train_label)

print('predict validataion data')
val_predict = logistic.predict_proba(val_data)
print(val_predict[:,1])
print(val_predict[:,1].shape)

val_loss = computing_loss.loss(val_label.values, val_predict[:,1])
print('validataion loss: %f' % (val_loss))

train_predict = logistic.predict_proba(train_data)
train_loss = computing_loss.loss(train_label.values, train_predict[:,1])
print('train loss: %f'%(train_loss))

print('predict test set')
test_predict = logistic.predict_proba(test_data)
computing_loss.write_res('res/logistic_regression_drop_str_data_fill_nan_mean_val.csv', test_id.values, test_predict[:,1])
