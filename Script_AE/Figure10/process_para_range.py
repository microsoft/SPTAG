import sys
import re
import csv


topkList = [0, 8, 64, 128]

avg_latency = []
accuracy = []

for i in topkList:
    templist_latency = []

    templist_accuracy = []

    templist_accuracy.append('')
    templist_latency.append('')

    fileName = sys.argv[1] + str(i)

    log_f = open(fileName)

    line_count = 0

    while True:
        line = log_f.readline()
        line_count+=1

        result_group = line.split()
        
        if len(result_group) > 2 and result_group[1] == "Total" and result_group[2] == "Vector":
            break

    while True:
        line = log_f.readline()
        line_count+=1

        if line == '':
            break

        result_group = line.split()
        
        if len(result_group) > 1 and result_group[0] == "Total" and result_group[1] == "Latency":
            line = log_f.readline()
            line_count+=1
            line = log_f.readline()
            line_count+=1
            result_group = line.split()
            templist_latency.append(float(result_group[1]))
        if len(result_group) > 2 and result_group[1] == "Recall10@10:":
            templist_accuracy.append(float(result_group[2]))

    accuracy.append(templist_accuracy)
    avg_latency.append(templist_latency)


with open("parameter_study_range.csv", 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerows(zip(avg_latency[0], accuracy[0], avg_latency[1], accuracy[1], avg_latency[2], accuracy[2], avg_latency[3], accuracy[3]))

