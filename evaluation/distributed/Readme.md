## 1. Prepare data: split into #machine partitions and copy data to different machines

```bash
python3 split_and_convert_data.py --data_file=/mnt/md0/qi/distributed_build/perftest_vector.bin.UInt8_999000000_128 --meta_file=/mnt/md0/qi/distributed_build/perftest_meta.bin.0_999000000 --metaidx_file=/mnt/md0/qi/distributed_build/perftest_metaidx.bin.0_999000000 --data_type=uint8 --vid_intype=int32 --batch_size=1000000 --partitions=3 --vid_outtype=int64 --output_dir=/mnt/md0/qi/distributed_build/
scp /mnt/md0/qi/distributed_build/*.bin.1 <hostname1>:/mnt/md0/qi/distributed_build/
scp /mnt/md0/qi/distributed_build/*.bin.2 <hostname2>:/mnt/md0/qi/distributed_build/
```
