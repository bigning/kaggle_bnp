import os,sys
import numpy as np
import pandas as pd
#from sklearn import linear_model
from sklearn import ensemble
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

test_data = test_data.loc[:, 'v1':'v131']

high_corr_columns = ["v8","v23","v25","v36","v37","v46","v51","v53","v54","v63","v73","v81","v82","v89","v92","v95","v105","v107","v108","v109","v116","v117","v118","v119","v123","v124","v128"]

### remove high correlate columns
for column_name in high_corr_columns:
    del train_data[column_name]
    del val_data[column_name]
    del test_data[column_name]

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
            for i in range(all_possible_value.size):
                m[all_possible_value[i]] = i
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
#train_data = drop_non_numberic(train_data, train_data)
#val_data = drop_non_numberic(train_data, val_data)
#test_data = drop_non_numberic(train_data, test_data)
categary_to_number()

## using mean value to replace nan
train_data = fill_nan(train_data, train_data)
val_data = fill_nan(train_data, val_data)
test_data = fill_nan(train_data, test_data)

print('training logistic regression model')
gbdt = sklearn.ensemble.GradientBoostingClassifier(learning_rate = 0.1, n_estimators=100, subsample=0.96, verbose = 2, max_depth = 11, max_features = 0.5)
gbdt.fit(train_data, train_label)

print('predict validataion data')
val_predict = gbdt.predict_proba(val_data)
print(val_predict[:,1])
print(val_predict[:,1].shape)

val_loss = computing_loss.loss(val_label.values, val_predict[:,1])
print('validataion loss: %f' % (val_loss))

train_predict = gbdt.predict_proba(train_data)
train_loss = computing_loss.loss(train_label.values, train_predict[:,1])
print('train loss: %f'%(train_loss))

print('predict test set')
test_predict = gbdt.predict_proba(test_data)
computing_loss.write_res('res/gbdt_rm_corr.csv', test_id.values, test_predict[:,1])
