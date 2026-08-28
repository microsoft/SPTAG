import sys
import re
import csv


insert = [1, 2, 4, 8]
append = [1, 2, 4]

threadlist = [11, 21, 41, 81, 82, 84]

fore_throughput = []
back_throughput = []

for i in threadlist:
    fileName = sys.argv[1] + str(i)

    log_f = open(fileName)

    while True:
        line = log_f.readline()

        if line == '':
            break

        result_group = line.split()
        
        if len(result_group) > 11 and result_group[1] == "Insert:" and result_group[7] == "sending":
            fore_throughput.append(float(result_group[10]))

        if len(result_group) > 11 and result_group[1] == "Insert:" and result_group[7] == "actuall":
            back_throughput.append(float(result_group[10].rstrip(',')))

with open("foreground_background.csv", 'w') as f:
    writer = csv.writer(f, delimiter=',')
    templist = []
    for i in range(0, 11):
        templist.append('')
    writer.writerow(templist)
    for i in range(0,4):
        templist = []
        templist.append('')
        templist.append(insert[i])
        if insert[i] != 8:
            templist.append(fore_throughput[i])
            templist.append(back_throughput[i])
            for j in range(0, 4):
                templist.append('')
            templist.append(insert[i])
            templist.append(fore_throughput[i+3])
            templist.append(back_throughput[i+3])
        else:
            for j in range(0,3):
                templist.append(fore_throughput[j+3])
                templist.append(back_throughput[j+3])
            for j in range(0, 3):
                templist.append('')
        writer.writerow(templist)


                
