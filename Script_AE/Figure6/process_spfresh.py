import sys
import re
import csv

log_f = open(sys.argv[1])

avg_latency = []
tail_latency = []
throughput = []
insert_avg_latency = []
RSS = []

line_count = 0

while True:
    try:
        line = log_f.readline()
        line_count+=1

        if line == '':
        	break

        result_group = line.split()
        
        if len(result_group) > 1 and result_group[1] == "Samppling":
            done = False
            while not done:
                line = log_f.readline()
                result_group = line.split()
                if len(result_group) > 4 and result_group[4] == "RSS":
                    RSS.append(int(result_group[16]))
                elif len(result_group) > 7 and result_group[7] == "AvgQPS:":
                    throughput.append(float(result_group[8].rstrip('.')))
                elif len(result_group) > 1 and result_group[0] == "Total" and result_group[1] == "Latency":
                    line = log_f.readline()
                    line = log_f.readline()
                    result_group = line.split()
                    avg_latency.append(float(result_group[1]))
                    tail_latency.append(float(result_group[6]))
                    done = True


    except AttributeError:
        print('Error when processing [at line]: {}'.format(line_count))
        break

log_f.close()
log_f = open(sys.argv[1])

while True:
    try:
        line = log_f.readline()
        line_count+=1

        if line == '':
        	break

        result_group = line.split()
        
        if len(result_group) > 2 and result_group[1] == "Insert" and result_group[2] == "Latency":
            line = log_f.readline()
            line_count+=1
            result_group = line.split()
            read = False
            while not read:
                if len(result_group) <= 1:
                    line = log_f.readline()
                    line_count+=1
                    result_group = line.split()
                    continue
                try:
                    data = float(result_group[1])
                except ValueError:
                    line = log_f.readline()
                    line_count+=1
                    result_group = line.split()
                    continue
                read = True
            insert_avg_latency.append(float(result_group[1]))


    except AttributeError:
        print('Error when processing [at line]: {}'.format(line_count))
        break
    except ValueError:
        print('Error when processing [at line]: {}: {}'.format(line_count , line))
        break

avg_latency_search = []
avg_latency_search.append('Search')
avg_latency_search.append('Avg Latency')
for i in range(0, len(avg_latency)):
    avg_latency_search.append(avg_latency[i])

tail_latency_search = []
tail_latency_search.append(' ')
tail_latency_search.append('Tail Latency')
for i in range(0, len(tail_latency)):
    tail_latency_search.append(tail_latency[i])

throughput_search = []
throughput_search.append('')
throughput_search.append('Throughput')
for i in range(0, len(throughput)):
    throughput_search.append(throughput[i])

RSS_search = []
RSS_search.append('')
RSS_search.append('RSS')
for i in range(0, len(RSS)):
    RSS_search.append(RSS[i]/1024)

insert_avg_latency_search = []
insert_avg_latency_search.append('Insert')
insert_avg_latency_search.append('Avg Latency')
for i in range(0, len(insert_avg_latency)):
    insert_avg_latency_search.append(insert_avg_latency[i])
    
for i in range(len(insert_avg_latency), len(RSS)):
    insert_avg_latency_search.append('')


with open(sys.argv[2], 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerows(zip(avg_latency_search, tail_latency_search, throughput_search, RSS_search, insert_avg_latency_search))