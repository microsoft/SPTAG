cp /home/sosp/store_spacev100m/data/indexloader_spann.ini /home/sosp/data/store_spacev100m/indexloader.ini 
PCI_ALLOWED="c636:00:00.0" SPFRESH_SPDK_USE_SSD_IMPL=1 SPFRESH_SPDK_CONF=/home/sosp/SPFresh/bdev.json SPFRESH_SPDK_BDEV=Nvme0n1 sudo -E /home/sosp/SPFresh/Release/spfresh /home/sosp/data/store_spacev100m/|tee log_overall_performance_spacev_spann.log
python process_spann.py log_overall_performance_spacev_spann.log overall_performance_spacev_spann_result.csv

mkdir spann_result
mv /home/sosp/result_overall_spacev_spann* spann_result

resultnamePrefix=/spann_result/
i=-1
for FILE in `ls -v1 ./spann_result/`
do
    if [ $i -eq -1 ];
    then
        /home/sosp/SPFresh/Release/usefultool -CallRecall true -resultNum 10 -queryPath /home/sosp/data/spacev_data/query.i8bin -searchResult $PWD$resultnamePrefix$FILE -truthType DEFAULT -truthPath /home/sosp/data/spacev_data/msspacev-100M -VectorPath /home/sosp/data/spacev_data/spacev200m_base.i8bin --vectortype int8 -d 100 -f DEFAULT |tee log_spann_$i
    else
        /home/sosp/SPFresh/Release/usefultool -CallRecall true -resultNum 10 -queryPath /home/sosp/data/spacev_data/query.i8bin -searchResult $PWD$resultnamePrefix$FILE -truthType DEFAULT -truthPath /home/sosp/data/spacev_data/spacev100m_update_truth_after$i -VectorPath /home/sosp/data/spacev_data/spacev200m_base.i8bin --vectortype int8 -d 100 -f DEFAULT |tee log_spann_$i
    fi
    let "i=i+1"
done