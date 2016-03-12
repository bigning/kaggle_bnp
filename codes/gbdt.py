import os,sys
import numpy as np
import pandas as pd
#from sklearn import linear_model
from sklearn import ensemble
import sklearn
import computing_loss
from optparse import OptionParser

parser = OptionParser()
parser.add_option('-l', '--learning_rate', dest='learning_rate', action='store', type='float',default=0.1)
parser.add_option('-n', '--n_estimators', dest='n_estimators', action='store', type='int',default=100)
parser.add_option('-d', '--max_depth', dest='max_depth', action='store', type='int',default=8)
parser.add_option('-f', '--save_name', dest='save_name', action='store', type='string')
(opt, args) = parser.parse_args()
if len(args)!=0:
    parser.print_help()
    sys.exit()
    
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
gbdt = sklearn.ensemble.GradientBoostingClassifier(learning_rate = opt.learning_rate, n_estimators=opt.n_estimators, subsample=0.8, verbose = 2, max_depth = opt.max_depth, max_features = 0.7)
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
computing_loss.write_res('res/'+opt.save_name, test_id.values, test_predict[:,1])
