#pragma once
#include <deque>
#include <mutex>

#include "rabitqlib/utils/hashset.hpp"

namespace rabitqlib {
class VisitedListPool {
    std::deque<HashBasedBooleanSet*> pool_;
    std::mutex poolguard_;
    size_t numelements_;

   public:
    VisitedListPool(size_t initpoolsize, size_t max_elements) {
        numelements_ = max_elements / 10;
        for (size_t i = 0; i < initpoolsize; i++) {
            pool_.push_front(new HashBasedBooleanSet(numelements_));
        }
    }

    HashBasedBooleanSet* get_free_vislist() {
        HashBasedBooleanSet* rez;
        {
            std::unique_lock<std::mutex> lock(poolguard_);
            if (pool_.size() > 0) {
                rez = pool_.front();
                pool_.pop_front();
            } else {
                rez = new HashBasedBooleanSet(numelements_);
            }
        }
        rez->clear();
        return rez;
    }

    void release_vis_list(HashBasedBooleanSet* vl) {
        std::unique_lock<std::mutex> lock(poolguard_);
        pool_.push_front(vl);
    }

    ~VisitedListPool() {
        while (pool_.size() > 0) {
            HashBasedBooleanSet* rez = pool_.front();
            pool_.pop_front();
            ::delete rez;
        }
    }
};
}  // namespace rabitqlib