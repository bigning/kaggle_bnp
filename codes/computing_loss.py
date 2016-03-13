import os,sys
import math

def loss(target, output):
    res = 0
    for i in range(target.size):
        if target[i] == 1:
            res += -math.log(output[i] + 0.00000000001)
        else:
            res += -math.log(1 - output[i] + 0.00000000001)
    res /= target.size
    return res

def write_res(filename, test_id, prob):
    f = open(filename, 'w')
    f.writelines('ID,PredictedProb\n')
    for i in range(test_id.size):
        f.writelines(('%d,%f\n'%( test_id[i], prob[i])))

    f.close()

