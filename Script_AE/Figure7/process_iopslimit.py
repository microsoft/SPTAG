import sys
import re
import csv


process_list = [1, 2, 4, 8, 10, 12]


throughput = []
KIOPS = []

throughput.append('')
KIOPS.append('')

line_count = 0

for i in process_list:
    log_f = open(sys.argv[1] + str(i))
    while True:
        line = log_f.readline()
        line_count+=1

        if line == '':
            break

        result_group = line.split()
        
        if len(result_group) > 7 and result_group[7] == "AvgQPS:":
            throughput.append(float(result_group[8].rstrip('.')))
            while result_group[0] != 'IOPS:':
                line = log_f.readline()
                line_count+=1
                result_group = line.split()
            KIOPS.append(float((result_group[1].rstrip('k')))*1000)
            break

process_list_search = []
process_list_search.append('')
process_list_search += process_list

batch = []
batch.append('')
for i in range(0, 6):
    batch.append(i)

print(KIOPS)

with open("IOPS_limit.csv", 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerows(zip(batch, throughput, KIOPS, process_list_search))