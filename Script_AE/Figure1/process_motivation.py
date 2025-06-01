import sys
import re
import csv

process_list = ["log_static.log", "log_nolimit.log"]

percentile = ['', 0.5, 0.9, 0.95, 0.99, 0.999, 1]

percentile_baseline = []

avg_latency = []
accuracy = []

last_data_group = []

for fileName in process_list:
    
    log_f = open(fileName)

    templist_latency = []

    templist_accuracy = []

    templist_accuracy.append('')
    templist_latency.append('')

    line_count = 0

    if fileName == "log_nolimit.log":

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
                last_data_group = result_group
            if len(result_group) > 2 and result_group[1] == "Recall10@10:":
                templist_accuracy.append(float(result_group[2]))
    else:
        while True:
            line = log_f.readline()
            line_count+=1

            if line == '':
                break

            result_group = line.split()

            if len(result_group) > 3 and result_group[1] == "Updating" and result_group[2] == "numThread:":
                break
            
            if len(result_group) > 1 and result_group[0] == "Total" and result_group[1] == "Latency":
                line = log_f.readline()
                line_count+=1
                line = log_f.readline()
                line_count+=1
                result_group = line.split()
                templist_latency.append(float(result_group[1]))
                last_data_group = result_group
            if len(result_group) > 2 and result_group[1] == "Recall10@10:":
                templist_accuracy.append(float(result_group[2]))

    accuracy.append(templist_accuracy)
    avg_latency.append(templist_latency)
    temp_baseline = []
    temp_baseline.append('')
    for i in range(2, 8):
        temp_baseline.append(float(last_data_group[i]))
    percentile_baseline.append(temp_baseline)

with open("motivation1.csv", 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerows(zip(avg_latency[0], accuracy[0], avg_latency[1], accuracy[1]))

with open("motivation.csv", 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerows(zip(percentile_baseline[0], percentile, percentile_baseline[1], percentile))
