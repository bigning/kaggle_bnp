import os,sys

f = open('../data/sample_submission.csv')
lines = f.readlines()
f.close()
f = open('all_1.csv', 'w')
f.writelines(lines[0])
lines = lines[1:]
for line in lines:
    line_arr = line.split(',')
    new_line = line_arr[0] + ',0.76\n'
    f.writelines(new_line)
    
f.close()
