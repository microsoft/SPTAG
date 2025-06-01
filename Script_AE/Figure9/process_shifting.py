import sys
import re
import csv

process_list = ["log_inplace.log", "log_static.log", "log_noreassign.log", "log_split+reassign.log"]

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

    if fileName != "log_static.log":

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

with open("parameter_study_shifting.csv", 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerows(zip(avg_latency[0], accuracy[0], avg_latency[1], accuracy[1], avg_latency[2], accuracy[2], avg_latency[3], accuracy[3]))
