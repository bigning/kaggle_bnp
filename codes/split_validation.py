import os,sys
import random

validation_ratio = 1.0/6.0

original_train_file = open('../data/train.csv')
original_train_data = original_train_file.readlines()
header_row = original_train_data[0]
original_train_data = original_train_data[1:]
original_train_file.close()

original_train_num = len(original_train_data)

validation_num = int(validation_ratio * float(original_train_num))

random.shuffle(original_train_data)

validation_data = original_train_data[:validation_num]
train_data = original_train_data[validation_num:]

f = open('../data/new_train.csv', 'w')
f.writelines(header_row)
for line in train_data:
    f.writelines(line)
f.close()

f = open('../data/validation.csv', 'w')
f.writelines(header_row)
for line in validation_data:
    f.writelines(line)
f.close()
