import sys
import re
import csv

log_f = open(sys.argv[1])

avg_latency = []
tail_latency = []
throughput = []
insert_avg_latency = []
RSS = []
KIOPS = []

avg_latency_batch = []
tail_latency_batch = []
throughput_batch = []
RSS_batch = []
KIOPS_batch = []


number_point_per_batch = 3

last_process = 0

line_count = 0

while True:
    try:
        line = log_f.readline()
        line_count+=1

        if line == '':
        	break

        result_group = line.split()
        
        if len(result_group) > 1 and result_group[1] == "Samppling":
            this_process = int(result_group[3]) % 10000000
            if this_process < last_process:

                batch_len = len(avg_latency_batch)
                duration = int(batch_len) / number_point_per_batch
                # print(batch_len)
                for i in range(0, number_point_per_batch):
                    avg_latency.append(float(avg_latency_batch[0+i*duration]))
                    tail_latency.append(float(tail_latency_batch[0+i*duration]))
                    throughput.append(float(throughput_batch[0+i*duration]))
                    RSS.append(RSS_batch[0+i*duration])
                    KIOPS.append(KIOPS_batch[0+i*duration])

                avg_latency_batch = []
                tail_latency_batch = []
                throughput_batch = []
                RSS_batch = []
            
            if this_process != 0:
                done = False
                while not done:
                    line = log_f.readline()
                    line_count+=1
                    result_group = line.split()
                    if len(result_group) > 4 and result_group[4] == "RSS":
                        RSS_batch.append(int(result_group[16]))
                    elif len(result_group) > 7 and result_group[7] == "AvgQPS:":
                        throughput_batch.append(float(result_group[8].rstrip('.')))
                    elif len(result_group) > 1 and result_group[0] == "Total" and result_group[1] == "Latency":
                        line = log_f.readline()
                        line_count+=1
                        line = log_f.readline()
                        line_count+=1
                        result_group = line.split()
                        read = False
                        while not read:
                            try:
                                data = float(result_group[1])
                            except ValueError:
                                line = log_f.readline()
                                line_count+=1
                                result_group = line.split()
                                continue
                            read = True
                        avg_latency_batch.append(float(result_group[1]))
                        tail_latency_batch.append(float(result_group[5]))
                        done = True
                line = log_f.readline()
                line_count+=1
                result_group = line.split()
                while result_group[0] != 'IOPS:':
                    line = log_f.readline()
                    line_count+=1
                    result_group = line.split()
                KIOPS_batch.append(float((result_group[1].rstrip('k'))))

                
            last_process = this_process

    except AttributeError:
        print('Error when processing [at line]: {}'.format(line_count))
        break
    except ValueError:
        print('Error when processing [at line]: {}: {}'.format(line_count , line))
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

if this_process != 0:
    batch_len = len(avg_latency_batch)
    duration = int(batch_len) / number_point_per_batch
    print(batch_len)

    for i in range(0, number_point_per_batch):
        avg_latency.append(float(avg_latency_batch[0+i*duration]))
        tail_latency.append(float(tail_latency_batch[0+i*duration]))
        throughput.append(float(throughput_batch[0+i*duration]))
        RSS.append(RSS_batch[0+i*duration])
        KIOPS.append(KIOPS_batch[0+i*duration])

step = 0.25
batch = []
batch.append('SPFresh')
batch.append('Batch')
for i in range(0,20):
    batch.append(i+step)
    batch.append(i+step*2)
    batch.append(i+step*3)

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

tail_latency_search = []
tail_latency_search.append('')
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
for i in range(0, len(insert_avg_latency)):
    insert_avg_latency_search.append(insert_avg_latency[i])
    
for i in range(len(insert_avg_latency), len(RSS)):
    insert_avg_latency_search.append('')

KIOPS_search = []
KIOPS_search.append('')
KIOPS_search.append('KIOPS')
for i in range(0, len(KIOPS)):
    KIOPS_search.append(KIOPS[i])

temp_search = []
temp_search.append('')
temp_search.append('')
for i in range(0, len(KIOPS)):
    temp_search.append('')

with open('stress_test.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',')
    for line in zip(batch, temp_search, avg_latency_search, tail_latency_search, throughput_search, RSS_search, insert_avg_latency_search, temp_search, temp_search, temp_search, KIOPS_search):
        writer.writerow(line)

batch_new = []
batch_new.append('SPFresh')
batch_new.append('Batch')
for i in range(0,20):
    batch_new.append(i)

insert_avg_latency_new = []
insert_avg_latency_new.append('Insert')
insert_avg_latency_new.append('Avg Latency')
for i in range(0, len(insert_avg_latency)):
    insert_avg_latency_new.append(insert_avg_latency[i])

temp_search_new = []
temp_search_new.append('')
temp_search_new.append('')
for i in range(0, 20):
    temp_search_new.append('')

# print(KIOPS_search)

with open('stress_test2.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerows(zip(batch_new, insert_avg_latency_new, temp_search_new, temp_search_new))

