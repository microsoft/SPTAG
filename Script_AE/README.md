# **Reproduce All Experiment Results(Result Reproduced)**

In this folder, we provide scripts for reproducing figures in our paper. A Standard_L16s_v3 instance is needed to reproduce all the results.

The name of each script corresponds to the number of each figure in our paper. Since some of the scripts in this directory take a long computing time (as we mark below), we strongly recommend you create a tmux session to avoid script interruption due to network instability.

## **Dataset and Benchmark**
> **For artifact evaluation, we strongly recommend the reviewers to use those pre-downloaded and built dataset and benchmark in the ~/data/ folder on the provided Lsv3 instance, since the provided index and dataset requires lots of computation resource and takes about two weeaks to generate these data in a 80-core machine (we build these data offline)**

To  generate data, you can refer to [./Script_AE/iniFile/README.md](./Script_AE_iniFile).

## **Additional Setup**
here is additional setup for evaluation

### **How to check current state**
> if we can not see the /dev/nvme0n1 after the following command, it means this disk has been taken over by SPDK
> Using SPDK will bind the disk to SPDK and the /dev/nvme0n1 will not be shown after "lsblk"
```bash
lsblk
```

> If the device has been binded to SPDK, the following command can reset
```bash
sudo ./SPFresh/ThirdParty/spdk/scripts/setup.sh reset
```

### **SPFresh & SPANN+**
> Since we use SPDK to build our storage, we need to bind PCI dev to SPDK
```bash
sudo nvme format /dev/nvme0n1
sudo ./SPFresh/ThirdParty/spdk/scripts/setup.sh
cp bdev.json /home/sosp/SPFresh/
```

### **DiskANN**
> DiskANN Baseline use the filesystem provided by kernel to maintain its on-disk file, so if we want to run DiskANN, we need to release the disk from SPDK
```bash
sudo ./SPFresh/ThirdParty/spdk/scripts/setup.sh reset
```

> Prepare DiskANN for evaluation
```bash
sudo mkfs.ext4 /dev/nvme0n1
sudo mount /dev/nvme0n1 /home/sosp/testbed
sudo chmod 777 /home/sosp/testbed
```

> If you want to switch Evaluation from DiskANN to SPFresh, you must umount the filesystem at first and follow the above SPDK command
```bash
sudo umount /home/sosp/testbed
sudo nvme format /dev/nvme0n1
sudo ./SPFresh/ThirdParty/spdk/scripts/setup.sh
```

## **Start to Run**

> all files for running is already set in those folders in Azure VM for AE.

### **Motivation (Figure 1)**
> It takes about 22 minutes
```bash
bash motivation.sh
```
> Plot the result
```bash
bash plot_motivation_result.sh
```

### **Overall Performance (Figure 6)**
> It takes about 6 days to reproduce this figure
#### **SPFresh**
> This takes about 43 hours and 50 minutes
```bash
bash overall_spacev_spfresh.sh
```
#### **SPANN+**
> This takes about 43 hours and 30 minutes
```bash
bash overall_spacev_spann.sh
```
#### **DiskANN**
> Before running, move DiskANN Index to the disk (if binded by SPDK, First reset and mkfs (follow the instruction above))
```bash
cp -r /home/sosp/data/store_diskann_100m /home/sosp/testbed
```
> This takes about 35 hours and 44 minutes
```bash
bash overall_spacev_diskann.sh
```
#### **Plot the result of the baselines**
```bash
bash plot_overall_result.sh
```

### **Disk IOPS Limitation (Figure 7)**
> This takes about 5 minutes
```bash
bash iops_limitation.sh
```
> Plot the result
```bash
bash plot_iops_result.sh
```

### **Stress Test (Figure 8)**
> This takes about 35 hours and 26 minutes
```bash
bash stress_spfresh.sh
```
> Plot the result
```bash
bash plot_stress_result.sh
```

### **Data Shifting Micro-benchmark (Figure 9)**
> This takes about 1 hours and 10 minutes
```bash
bash data_shifting.sh
```
> Plot the result
```bash
bash plot_shifting_result.sh
```
### **Parameter Study: Reassign Range (Figure 10)**
> This takes about 1 hours and 30 minutes
```bash
bash parameter_study_range.sh
```
> Plot the result
```bash
bash plot_range_result.sh
```
### **Foreground Scalability (Figure 11)**
> This takes about 11 minutes
```bash
bash parameter_study_balance.sh
```
> Plot the result
```bash
bash plot_balance_result.sh
```



