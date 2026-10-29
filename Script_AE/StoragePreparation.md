# Dedicated-machine storage preparation

[Historical artifact guide](README.md) |
[Current installation](../docs/LinuxInstallation.md)

**Danger: formatting, filesystem creation, and SPDK device rebinding can
destroy data or disconnect storage.** This procedure is for an explicitly
allocated experimental device only. It is not required for compiling SPTAG,
running the quick start, or using an existing FILEIO filesystem.

## Separate build from storage administration

Compile SPDK and its dependencies according to the pinned dependency's
instructions on Linux. Dependency installation may require administrator
approval; it does not authorize disk formatting or rebinding. The retired
`build_spresh.sh init_env` mode no longer performs those operations.

## Required checks before any destructive operation

The machine owner/operator must:

1. Identify the device by persistent identity (serial/model/WWN), capacity,
   PCI address, and current ownership, not a copied `/dev/nvme0n1` name.
2. Confirm that it is neither the OS disk nor shared storage, mounted storage,
   swap, RAID/LVM, or an active workload's dependency.
3. Back up required data and receive explicit authorization to erase the
   identified device.
4. Stop its workloads and review the exact vendor commands independently.
5. Record the device identity, intended SPDK binding or filesystem, mountpoint,
   and permission model.

Read-only inspection may include:

```bash
lsblk -o NAME,SIZE,MODEL,SERIAL,FSTYPE,MOUNTPOINTS
findmnt
```

An absent kernel block-device entry does not by itself prove that SPDK owns a
device; inspect PCI binding and service state as well.

## SPDK versus filesystem experiments

SPDK uses userspace access and an explicit PCI allowlist/bdev configuration.
Filesystem experiments such as DiskANN need the device bound back to a suitable
kernel driver and a separately managed filesystem/mountpoint. Reformatting
between them is not a reversible "reset".

Use the pinned SPDK and NVMe tooling documentation with the verified device
identity. This guide intentionally supplies **no copy-and-paste format or
wipe command** and does not run any storage administration automatically.
Do not use world-writable permissions as a general installation fix.

## Output paths

Choose a new, empty run directory under the authorized mountpoint and a separate
checkpoint directory. Pass them explicitly as `SPTAG_EXPERIMENT_SSD_PATH` and
`SPTAG_EXPERIMENT_CHECKPOINT_PATH` for the legacy driver. Its `build_index`
mode refuses a nonempty SSD output directory rather than deleting prior data.
Review remaining paper-specific configuration and any SPDK environment values
before opting into a legacy mode.

Select `SPTAG_EXPERIMENT_STORAGE=SPDKIO` only for an authorized prepared device.
The SPDK update mode requires explicit `PCI_ALLOWED`, `SPFRESH_SPDK_CONF`, and
`SPFRESH_SPDK_BDEV`; no personal PCI address is provided as a default. FILEIO
updates do not invoke `sudo` or inject SPDK settings. Existing result files
cause refusal rather than automatic deletion.
