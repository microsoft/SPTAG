# Historical SPFresh paper artifact evaluation

[Evaluation index](../evaluation/README.md) |
[Current installation](../docs/LinuxInstallation.md)

This is a paper-era, dedicated-machine workflow. The original "Result
Reproduced" label described artifact evaluation at that time, not a validation
of this branch or a guarantee for a new machine. Hosts, paths, toolchains, and
runtime estimates below are historical.

**Do not run this workflow on a shared machine or a disk containing data.**
Disk formatting, rebinding, and filesystem creation have been separated into
the [dedicated storage preparation procedure](StoragePreparation.md); they are
never part of ordinary installation.

The root `build_spresh.sh init_env` mode is retired. Other legacy modes require
`SPTAG_ENABLE_LEGACY_EXPERIMENTS=1`, `SPTAG_EXPERIMENT_SSD_PATH`, and
`SPTAG_EXPERIMENT_CHECKPOINT_PATH`. This is an explicit opt-in, not certification
that the historical experiments work unchanged. `build_index` now requires an
empty output directory rather than deleting existing contents.

In this folder, we provide scripts for reproducing figures in our paper. A Standard_L16s_v3 instance is needed to reproduce all the results.

The name of each script corresponds to the number of each figure in our paper. Since some of the scripts in this directory take a long computing time (as we mark below), we strongly recommend you create a tmux session to avoid script interruption due to network instability.

## **Dataset and Benchmark**
> **For artifact evaluation, we strongly recommend the reviewers to use those pre-downloaded and built dataset and benchmark in the ~/data/ folder on the provided Lsv3 instance, since the provided index and dataset requires lots of computation resource and takes about two weeaks to generate these data in a 80-core machine (we build these data offline)**

To generate data, refer to the [dataset configuration guide](iniFile/README.md).

## Storage setup is separate

Have the machine owner complete [storage preparation](StoragePreparation.md)
before selecting an SPDK or filesystem-based experiment. Switching between
SPFresh/SPANN+ and DiskANN may require changing device ownership and recreating
storage, which destroys existing contents. No shared device name is a safe
default.

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


