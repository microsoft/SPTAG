cp /home/sosp/data/store_spacev100m/indexloader_iopslimit.ini /home/sosp/data/store_spacev100m/indexloader.ini 
SearchThreadNumLine="107c SearchThreadNum="
loaderPath="/home/sosp/data/store_spacev100m/indexloader.ini "
storePath="/home/sosp/data/store_spacev100m"
logPath="log_searchthread"

for i in 1 2 4 8 10 12
do
    newSearchThreadNumLine=$SearchThreadNumLine$i
    sed -i "$newSearchThreadNumLine" $loaderPath
    PCI_ALLOWED="c636:00:00.0" SPFRESH_SPDK_USE_SSD_IMPL=1 SPFRESH_SPDK_CONF=/home/sosp/SPFresh/bdev.json SPFRESH_SPDK_BDEV=Nvme0n1 sudo -E /home/sosp/SPFresh/Release/spfresh $storePath |tee $logPath$i
done