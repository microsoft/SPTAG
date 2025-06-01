setname="6c VectorPath=spacev100m_update_set"
truthname="18c TruthPath=spacev100m_update_truth"
deletesetname="spacev100m_update_set"
reservesetname="spacev100m_update_reserve"
currentsetname="spacev100m_update_current"
for i in {0..99}
do
    /home/sosp/SPFresh/Release/usefultool -GenTrace true --vectortype int8 --VectorPath /home/sosp/data/spacev_data/spacev200m_base.i8bin --filetype DEFAULT --UpdateSize 1000000 --BaseNum 100000000 --ReserveNum 100000000 --CurrentListFileName spacev100m_update_current --ReserveListFileName spacev100m_update_reserve --TraceFileName spacev100m_update_trace -NewDataSetFileName spacev100m_update_set -d 100 --Batch $i -f DEFAULT
    newsetname=$setname$i
    newtruthname=$truthname$i
    newdeletesetname=$deletesetname$i
    newreservesetname=$reservesetname$i
    newcurrentsetname=$currentsetname$i
    sed -i "$newsetname" genTruth.ini
    sed -i "$newtruthname" genTruth.ini
    /home/sosp/SPFresh/Release/ssdserving genTruth.ini
    /home/sosp/SPFresh/Release/usefultool -ConvertTruth true --vectortype int8 --VectorPath /home/sosp/data/spacev_data/spacev200m_base.i8bin --filetype DEFAULT --UpdateSize 1000000 --BaseNum 100000000 --ReserveNum 100000000 --CurrentListFileName spacev100m_update_current --ReserveListFileName spacev100m_update_reserve --TraceFileName spacev100m_update_trace -NewDataSetFileName spacev100m_update_set -d 100 --Batch $i -f DEFAULT --truthPath spacev100m_update_truth --truthType DEFAULT --querySize 10000 --resultNum 100
    rm -rf $deletesetname
    rm -rf $newreservesetname
    rm -rf $newcurrentsetname
done