import sys
import re
import csv
line_count = 0

# spfresh 0, spann 1, diskann 2
accuracy_list=[]

for j in range(1, 4):
    templist = []
    templist.append('')
    templist.append('Accuracy')
    for i in range(-1, 100):
        fileName = sys.argv[j] + str(i)
        log_f = open(fileName)
        while True:
            line = log_f.readline()
            line_count+=1

            if line == '':
                break

            result_group = line.split()
            if len(result_group) > 1 and result_group[1] == "Recall10@10:":
                templist.append(float(result_group[2]))
    while len(templist) != 402:
        templist.append('')
        
    accuracy_list.append(templist)

# spfresh 3, spann 4, diskann 5

csv_reader_spfresh = csv.reader(open(sys.argv[4]))
csv_reader_spann = csv.reader(open(sys.argv[5]))
csv_reader_diskann = csv.reader(open(sys.argv[6]))

step = 0.2
batch = []
batch.append('')
batch.append('Batch')
for i in range(0,100):
    batch.append(i+step)
    batch.append(i+step*2)
    batch.append(i+step*3)
    batch.append(i+step*4)


with open('overall_performance_spacev.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',')
    i = 0
    for row1, row2, row3 in zip(csv_reader_spfresh, csv_reader_diskann, csv_reader_spann):
        line = []
        line.append(batch[i])
        line.append('')
        line += row1
        line.append('')
        line.append(accuracy_list[0][i])
        line.append('')
        line += row2
        line.append('')
        line.append(accuracy_list[2][i])
        line.append('')
        line += row3 
        line.append('')
        line.append(accuracy_list[1][i])
        writer.writerow(line)
        i+=1

batch = []
batch.append('')
batch.append('Batch')
for i in range(0,101):
    batch.append(i)

csv_reader_spfresh = csv.reader(open(sys.argv[4]))
csv_reader_spann = csv.reader(open(sys.argv[5]))
csv_reader_diskann = csv.reader(open(sys.argv[6]))

with open('overall_performance_spacev2.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',')
    i = 0
    for row1, row2, row3 in zip(csv_reader_spfresh, csv_reader_diskann, csv_reader_spann):
        if i == 101:
            break
        line = []
        line.append(batch[i])
        line.append(row1[4])
        line.append('')
        line.append(accuracy_list[0][i])
        line.append(row2[4])
        line.append('')
        line.append(accuracy_list[2][i])
        line.append(row3[4])
        line.append('')
        line.append(accuracy_list[1][i])
        writer.writerow(line)
        i+=1

        






