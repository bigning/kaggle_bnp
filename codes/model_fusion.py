res_file_names = ['./res/gbdt_rm_corr_0.47171.csv', './res/gbdt_0.4697.csv', './res/gbdt.csv']
weights = [1.0/3.0, 1.0/3.0, 1.0/3.0]

new_res = {}
index = 0
new_str = ''
for name in res_file_names:
    name_arr = name.split('/')
    new_str = new_str + '_' + name_arr[len(name_arr) - 1]

new_res_file = open('./res/' + new_str + '.csv', 'w')
for file_name in res_file_names:
    f = open(file_name)
    line = f.readline()
    if index == 0:
        new_res_file.writelines(line)
    lines = f.readlines()

    for i,line in enumerate(lines):
        if i % 5000 == 0:
            print(('%d_%d')%(index, i))
        line = line.strip('\n')
        line_arr = line.split(',')
        number = line_arr[0]
        if new_res.has_key(number):
            new_res[number] = new_res[number] + weights[index] * float(line_arr[1])
        else:
            new_res[number] = weights[index] * float(line_arr[1])
    f.close()
    index = index + 1

for key in new_res:
    new_res_file.writelines(key + ','+str(new_res[key]) + '\n')
