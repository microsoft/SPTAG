loaderPath="/home/sosp/data/store_sift1m/indexloader.ini"
storePath="/home/sosp/data/store_sift1m"
InsertLine="109c InsertThreadNum="
AppendLine="110c AppendThreadNum="
DeleteQPSLine="122c DeleteQPS="
Insert=1
Append=1
DeleteQPS=1000
logPath="log_"
newDeleteQPS=0


for i in {1..4}
do
    let 'newDeleteQPS=Insert*DeleteQPS'
    newInsertLine=$InsertLine$Insert
    newAppendLine=$AppendLine$Append
    newDeleteQPSLine=$DeleteQPSLine$newDeleteQPS
    sed -i "$newInsertLine" $loaderPath
    sed -i "$newAppendLine" $loaderPath
    sed -i "$newDeleteQPSLine" $loaderPath
    PCI_ALLOWED="c636:00:00.0" SPFRESH_SPDK_USE_SSD_IMPL=1 SPFRESH_SPDK_CONF=/home/sosp/SPFresh/bdev.json SPFRESH_SPDK_BDEV=Nvme0n1 sudo -E /home/sosp/SPFresh/Release/spfresh $storePath |tee $logPath$Insert$Append
    let 'Insert=Insert*2'
done

let 'Insert=8'

for i in {1..2}
do
    let 'Append=Append*2'
    let 'newDeleteQPS=Insert*DeleteQPS'
    newInsertLine=$InsertLine$Insert
    newAppendLine=$AppendLine$Append
    newDeleteQPSLine=$DeleteQPSLine$newDeleteQPS
    sed -i "$newInsertLine" $loaderPath
    sed -i "$newAppendLine" $loaderPath
    sed -i "$newDeleteQPSLine" $loaderPath
    PCI_ALLOWED="c636:00:00.0" SPFRESH_SPDK_USE_SSD_IMPL=1 SPFRESH_SPDK_CONF=/home/sosp/SPFresh/bdev.json SPFRESH_SPDK_BDEV=Nvme0n1 sudo -E /home/sosp/SPFresh/Release/spfresh $storePath |tee $logPath$Insert$Append
done