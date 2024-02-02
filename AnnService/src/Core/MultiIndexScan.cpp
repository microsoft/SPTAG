#include "inc/Core/MultiIndexScan.h"

namespace SPTAG
{
    MultiIndexScan::MultiIndexScan() {}
    MultiIndexScan::MultiIndexScan(std::vector<std::shared_ptr<VectorIndex>> vectorIndices,
                                   std::vector<void*> p_targets,
                                   unsigned int k,
                                   float (*rankFunction)(std::vector<float>),
                                   bool useTimer,
                                   int termCondVal,
                                   int searchLimit
                                   ){
        // copy parameters
        this->fwdLUTs           = std::vector<std::shared_ptr<VectorIndex>>(vectorIndices);
        this->k                 = k;
        this->func              = rankFunction;
        // terminate related initialization
        this->useTimer          = useTimer;
        this->termCondVal       = termCondVal;
        this->searchLimit       = searchLimit;
        this->t_start           = std::chrono::high_resolution_clock::now();
        this->consecutive_drops = 0;
        // internal states
        // topK status
        this->pq                = std::priority_queue<pq_item, std::vector<pq_item>, pq_item_compare>();
        this->seenSet           = std::unordered_set<SizeType>();
        // output dataStrucure
        this->outputStk         = std::stack<pq_item>();
        this->terminate         = false;

        for ( int i = 0; i < fwdLUTs.size(); i++ ){
            indexIters.push_back(vectorIndices[i]->GetIterator(p_targets[i], true));
        }
        printf("(%zu, %zu, %zu)\n", fwdLUTs.size(), p_targets.size(), indexIters.size() );
    }
    float rankFunc(std::vector<float> in) {
        return (float)std::accumulate(in.begin(), in.end(), 0.0f);
    }

    float MultiIndexScan::WeightedRankFunc(std::vector<float> in) {
        float result = 0;
        for (int i = 0; i < in.size(); i++)
        {
            result += in[i] * weight[i];
        }
        return result;
    }

    void MultiIndexScan::Init(std::vector<std::shared_ptr<VectorIndex>> vectorIndices,
        std::vector<ByteArray> p_targets,
        std::vector<float> weight,
        unsigned int k,
        bool useTimer,
        int termCondVal,
        int searchLimit)
    {
        this->fwdLUTs = vectorIndices;
        this->k = k;
        this->func = nullptr;
        this->weight = weight;
        // terminate related initialization
        this->useTimer = useTimer;
        this->termCondVal = termCondVal;
        this->searchLimit = searchLimit;
        this->t_start = std::chrono::high_resolution_clock::now();
        this->consecutive_drops = 0;
        // internal states
        // topK status
        this->pq = std::priority_queue<pq_item, std::vector<pq_item>, pq_item_compare>();
        this->seenSet = std::unordered_set<SizeType>();
        // output dataStrucure
        this->outputStk = std::stack<pq_item>();
        this->terminate = false;

        for (int i = 0; i < fwdLUTs.size(); i++) {
            p_data_array.push_back(p_targets[i]);
            indexIters.push_back(vectorIndices[i]->GetIterator(p_data_array[i].Data(), true));
        }
        printf("(%zu, %zu, %zu)\n", fwdLUTs.size(), p_targets.size(), indexIters.size());
    }

    MultiIndexScan::~MultiIndexScan(){
        Close();
    }

	bool MultiIndexScan::Next(BasicResult& result)
	{
        int numCols = (int)indexIters.size();
        while ( !terminate ) {
            
            for ( int i = 0 ; i < numCols; i++ ) {
                auto result_iter = indexIters[i];
               // printf("probing index %d, %s\n", i, fwdLUTs[i]->GetIndexName().c_str());
                auto results = result_iter->Next(1);
                if (results->GetResultNum() == 0) {
                    printf("index %d no more result!! Terminating\n", i);                    
                    terminate = true;
                    break;
                }

                auto vid  = results->GetResult(0)->VID;
                auto dist = results->GetResult(0)->Dist;
                //we ignore meta for now:: auto meta = curr_result.Meta;

               // printf("vid = %lu, dist = %f from index %d\n", vid, dist, i);

                // insert into heap only if it has NOT been seen
                if ( seenSet.find(vid) == seenSet.end() ){
                    
                    std::vector<float> dists(numCols, 0);
                    dists[i] = dist;
                    
                    // lookup scores using foward index
                    for ( int j = 0; j < numCols; j++ ){
                        if ( i != j ){
                            dists[j] = fwdLUTs[j]->GetDistance(indexIters[j]->GetTarget(), vid);
                        }
                    }

                    // using UDF to calculate the score
                    float score;
                    if (func == nullptr)
                    {
                        score = WeightedRankFunc(dists);
                    }
                    else
                    {
                        score = func(dists);
                    }

                 //   printf("vid = %d, not seen! score = %f\n", vid, score);

                    // insert into heap only if it is smaller than the largest score
                    // in our ANN case, we keep the k smallest score/dists
                    if ( pq.size() == k && pq.top().first <= score ) {
                        //printf("%f >= largest score in heap %f, drop", score, pq.top().first);
                        consecutive_drops++;
                    } else {
                        consecutive_drops=0;
                        if ( pq.size() == k ) pq.pop(); // the largest score is squeezed out by the current score
                        pq.push(std::make_pair(score, vid));
                    }

                    // insert it into our seen set;
                    seenSet.insert(vid);
                }
            }
            
            // checkout terminate condition and check if we need to start stream out output
            // why we check term cond outside for loop? because we don't want to check that often~
            if ( useTimer ) {
                auto t_end = std::chrono::high_resolution_clock::now();
                terminate = (std::chrono::duration<double, std::milli>(t_end-t_start).count() >= termCondVal);
            } else if ( !useTimer && consecutive_drops >= termCondVal){
                terminate = true;
            } else if ( seenSet.size() >= searchLimit){
                terminate = true;
            }
        }
        // put back the output into the correct order, i.e. smallest score first
        if ( pq.size() > 0 ){
            while ( !pq.empty() ){
                outputStk.push(pq.top());
                pq.pop();
            }
        }

        // if no outputs
        if ( outputStk.size() == 0 ) return false;

        // pop-out outputs
        result.VID = outputStk.top().second;
        result.Dist = outputStk.top().first;        
        outputStk.pop();
        
        return true;
	}

	// Add end into destructor.
	void MultiIndexScan::Close()
	{
        for ( auto resultIter : indexIters ){
            resultIter->Close();
        }
	}

} // namespace SPTAG
