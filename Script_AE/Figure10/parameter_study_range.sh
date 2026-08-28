loaderPath="/home/sosp/data/store_sift_cluster/indexloader.ini"
storePath="/home/sosp/data/store_sift_cluster"
ReassignLine="118c ReassignK="
logPath="log_top"

cp /home/sosp/data/store_sift_cluster/indexloader_top64.ini /home/sosp/data/store_sift_cluster/indexloader.ini

for i in 0 8 64 128
do
    newReassignLine=$ReassignLine$i
    sed -i "$newReassignLine" $loaderPath
    PCI_ALLOWED="c636:00:00.0" SPFRESH_SPDK_USE_SSD_IMPL=1 SPFRESH_SPDK_CONF=/home/sosp/SPFresh/bdev.json SPFRESH_SPDK_BDEV=Nvme0n1 sudo -E /home/sosp/SPFresh/Release/spfresh $storePath |tee $logPath$i
done