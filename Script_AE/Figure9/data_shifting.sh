# noreassign/static/top64/inplace

# noreassign
cp /home/sosp/data/store_sift_cluster/indexloader_noreassign.ini /home/sosp/data/store_sift_cluster/indexloader.ini
PCI_ALLOWED="c636:00:00.0" SPFRESH_SPDK_USE_SSD_IMPL=1 SPFRESH_SPDK_CONF=/home/sosp/SPFresh/bdev.json SPFRESH_SPDK_BDEV=Nvme0n1 sudo -E /home/sosp/SPFresh/Release/spfresh /home/sosp/data/store_sift_cluster |tee log_noreassign.log

# static
PCI_ALLOWED="c636:00:00.0" SPFRESH_SPDK_USE_SSD_IMPL=1 SPFRESH_SPDK_CONF=/home/sosp/SPFresh/bdev.json SPFRESH_SPDK_BDEV=Nvme0n1 sudo -E /home/sosp/SPFresh/Release/spfresh /home/sosp/data/store_sift_cluster_2m |tee log_static.log

# inplace
cp /home/sosp/data/store_sift_cluster/indexloader_inplace.ini /home/sosp/data/store_sift_cluster/indexloader.ini
PCI_ALLOWED="c636:00:00.0" SPFRESH_SPDK_USE_SSD_IMPL=1 SPFRESH_SPDK_CONF=/home/sosp/SPFresh/bdev.json SPFRESH_SPDK_BDEV=Nvme0n1 sudo -E /home/sosp/SPFresh/Release/spfresh /home/sosp/data/store_sift_cluster |tee log_inplace.log

# top64
cp /home/sosp/data/store_sift_cluster/indexloader_top64.ini /home/sosp/data/store_sift_cluster/indexloader.ini
PCI_ALLOWED="c636:00:00.0" SPFRESH_SPDK_USE_SSD_IMPL=1 SPFRESH_SPDK_CONF=/home/sosp/SPFresh/bdev.json SPFRESH_SPDK_BDEV=Nvme0n1 sudo -E /home/sosp/SPFresh/Release/spfresh /home/sosp/data/store_sift_cluster |tee log_split+reassign.log