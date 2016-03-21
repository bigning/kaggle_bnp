import pandas as pd
import numpy as np
import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import ensemble
from sklearn.metrics import log_loss
from sklearn.naive_bayes import BernoulliNB
from optparse import OptionParser
import computing_loss

parser = OptionParser()
parser.add_option('-n', '--n_estimators', dest='n_estimators', action='store', type='int',default=750)
parser.add_option('-d', '--max_depth', dest='max_depth', action='store', type='int',default=40)
parser.add_option('-f', '--save_name', dest='save_name', action='store', type='string')
(opt, args) = parser.parse_args()
if len(args)!=0:
    parser.print_help()
    sys.exit()

def categary_to_number(column_name):
    if train.dtypes[column_name] == 'object':
        all_possible_value = []
        train_possible_value = np.unique(train[column_name])
        test_possible_value = np.unique(test[column_name])
        val_possible_value = np.unique(validation[column_name])
        all_possible_value = np.concatenate((train_possible_value, test_possible_value, val_possible_value))
        all_possible_value = np.unique(all_possible_value)
        m = {}
        for i in range(all_possible_value.size):
            m[all_possible_value[i]] = i
        train[column_name] = train[column_name].map(m)
        test[column_name] = test[column_name].map(m)
        validation[column_name] = validation[column_name].map(m)


def Binarize(columnName, df, features=None):
    df[columnName] = df[columnName].astype(str)
    if(features is None):
        features = np.unique(df[columnName].values)
    print(features)
    for x in features:
        df[columnName+'_' + x] = df[columnName].map(lambda y:
                                                    1 if y == x else 0)
    df.drop(columnName, inplace=True, axis=1)
    return df, features


def MungeData(train, test, validation):

    features = train.columns[2:]
    print(type(features))
    for col in features:
        if((train[col].dtype == 'object') and (col!="v22")):
            print(col)
            train, binfeatures = Binarize(col, train)
            test, _ = Binarize(col, test, binfeatures)
            validation , _ = Binarize(col, validation, binfeatures)
            nb = BernoulliNB()
            nb.fit(train[col+'_'+binfeatures].values, train.target.values)
            train[col] = \
                nb.predict_proba(train[col+'_'+binfeatures].values)[:, 1]
            test[col] = \
                            nb.predict_proba(test[col+'_'+binfeatures].values)[:, 1]
            validation[col] = \
                            nb.predict_proba(validation[col+'_'+binfeatures].values)[:, 1]
            train.drop(col+'_'+binfeatures, inplace=True, axis=1)
            test.drop(col+'_'+binfeatures, inplace=True, axis=1)
            validation.drop(col+'_'+binfeatures, inplace=True, axis=1)
            train[col] = train[col].astype(float)
            test[col] = test[col].astype(float)
            validation[col] = validation[col].astype(float)
    return train, test, validation


print('Load data...')
#train = pd.read_csv("../input/train.csv")
train = pd.read_csv("../data/new_train.csv")
#test = pd.read_csv("../input/test.csv")
test = pd.read_csv("../data/test.csv")
validation = pd.read_csv("../data/validation.csv")


train, test, validation = MungeData(train, test, validation)


target = train['target'].values
train = train.drop(['ID','target','v8','v23','v25','v31','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128'],axis=1)

id_test = test['ID'].values
test = test.drop(['ID','v8','v23','v25','v31','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128'],axis=1)

target_val= validation['target']
validation= validation.drop(['ID','target','v8','v23','v25','v31','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128'],axis=1)


print('Clearing...')
for (train_name, train_series), (test_name, test_series) in zip(train.iteritems(),test.iteritems()):
    if train_series.dtype == 'O':
        #for objects: factorize
        #but now we have -1 values (NaN)
        print(train_name)
        categary_to_number(train_name)
    else:
        #for int or float: fill NaN
        tmp_len = len(train[train_series.isnull()])
        if tmp_len>0:
            #print "mean", train_series.mean()
            train.loc[train_series.isnull(), train_name] = -999
        #and Test
        tmp_len = len(test[test_series.isnull()])
        if tmp_len>0:
            test.loc[test_series.isnull(), test_name] = -999

        tmp_len = len(validation[train_name].isnull())
        if tmp_len > 0:
            validation.loc[validation[train_name].isnull(), train_name] = -999

X_train = train
X_test = test
print('Training...')
extc = ExtraTreesClassifier(n_estimators=opt.n_estimators,max_features= 60,criterion= 'entropy',min_samples_split= 4,max_depth= opt.max_depth, min_samples_leaf= 2, n_jobs = -1, verbose = 2, random_state=3211254)

extc.fit(X_train,target)

print('Predict...')
y_pred = extc.predict_proba(X_test)
#print y_pred

val_pred = extc.predict_proba(validation)
print(val_pred)

val_loss = computing_loss.loss(target_val.values, val_pred[:,1])
print('validataion loss: %f' % (val_loss))

pd.DataFrame({"ID": id_test, "PredictedProb": y_pred[:,1]}).to_csv(opt.save_name,index=False)

