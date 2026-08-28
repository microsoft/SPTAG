/home/sosp/DiskANN_Baseline/build/tests/overall_performance int8 ~/data/spacev_data/spacev100m_base.i8bin 75 32 1.2 75 64 1.2 100000000 1 25 0 ~/testbed/store_diskann_100m/diskann_spacev_100m true false ~/data/spacev_data/spacev200m_base.i8bin ~/data/spacev_data/query.i8bin ~/data/truth 10 40 2 ~/data/spacev_data/spacev100m_update_trace 100 |tee log_overall_performance_spacev_diskann.log
python process_diskann.py log_overall_performance_spacev_diskann.log overall_performance_spacev_diskann_result.csv

mkdir diskann_result
mv /home/sosp/result_overall_spacev_diskann* diskann_result

resultnamePrefix=/diskann_result/
i=-1
for FILE in `ls -v1 ./diskann_result/`
do
    if [ $i -eq -1 ];
    then
        /home/sosp/SPFresh/Release/usefultool -CallRecall true -resultNum 10 -queryPath /home/sosp/data/spacev_data/query.i8bin -searchResult $PWD$resultnamePrefix$FILE -truthType DEFAULT -truthPath /home/sosp/data/spacev_data/msspacev-100M -VectorPath /home/sosp/data/spacev_data/spacev200m_base.i8bin --vectortype int8 -d 100 -f DEFAULT |tee log_diskann_$i
    else
        /home/sosp/SPFresh/Release/usefultool -CallRecall true -resultNum 10 -queryPath /home/sosp/data/spacev_data/query.i8bin -searchResult $PWD$resultnamePrefix$FILE -truthType DEFAULT -truthPath /home/sosp/data/spacev_data/spacev100m_update_truth_after$i -VectorPath /home/sosp/data/spacev_data/spacev200m_base.i8bin --vectortype int8 -d 100 -f DEFAULT |tee log_diskann_$i
    fi
    let "i=i+1"
done