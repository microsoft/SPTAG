import sys
import re
import csv

log_f = open(sys.argv[1])

avg_latency = []
tail_latency = []
throughput = []
insert_throughput = []
RSS = []

avg_latency_batch = []
tail_latency_batch = []
throughput_batch = []
RSS_batch = []

number_point_per_batch = 4

number_thread = 3

last_process = 0

line_count = 0

memory = 0

while True:
    try:
        line = log_f.readline()
        line_count+=1

        if line == '':
        	break

        result_group = line.split()

        if result_group[0] == "memory":
            memory = float(result_group[6])

        if result_group[0] != "Queries" and result_group[0] != "Inserted":
            pre_result_group = result_group
            continue
        
        if result_group[0] == "Queries":
            if int(result_group[2]) < last_process:
                # new batch
                batch_len = len(avg_latency_batch)
                duration = int(batch_len) / number_point_per_batch
                
                for i in range(0, number_point_per_batch):
                    avg_latency.append(float(avg_latency_batch[0+i*duration]))
                    tail_latency.append(float(tail_latency_batch[0+i*duration]))
                    throughput.append(float(throughput_batch[0+i*duration]))
                    RSS.append(RSS_batch[0+i*duration] / 1024 / 1024)

                avg_latency_batch = []
                tail_latency_batch = []
                throughput_batch = []
                RSS_batch = []

            avg_latency_batch.append(pre_result_group[2])
            tail_latency_batch.append(pre_result_group[7])
            throughput_batch.append(pre_result_group[1])
            RSS_batch.append(memory)
            last_process = int(result_group[2])
        
        if result_group[0] == "Inserted":
            insert_throughput.append(int(result_group[1]) / float(result_group[4].strip('s')) / number_thread)

    except AttributeError:
        print('Error when processing [at line]: {}'.format(line_count))
        break


batch_len = len(avg_latency_batch)
duration = int(batch_len) / number_point_per_batch

for i in range(0, number_point_per_batch):
    avg_latency.append(float(avg_latency_batch[0+i*duration]))
    tail_latency.append(float(tail_latency_batch[0+i*duration]))
    throughput.append(float(throughput_batch[0+i*duration]))
    RSS.append(RSS_batch[0+i*duration] / 1024 / 1024)

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
    RSS_search.append(RSS[i])

insert_avg_latency_search = []
insert_avg_latency_search.append('Insert')
insert_avg_latency_search.append('Avg Latency')
for i in range(0, len(insert_throughput)):
    insert_avg_latency_search.append( 1000000 / insert_throughput[i])
    
for i in range(len(insert_throughput), len(RSS)):
    insert_avg_latency_search.append('')

with open(sys.argv[2], 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerows(zip(avg_latency_search, tail_latency_search, throughput_search, RSS_search, insert_avg_latency_search))