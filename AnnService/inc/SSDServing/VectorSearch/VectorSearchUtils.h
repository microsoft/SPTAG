#pragma once

namespace SPTAG {
    namespace SSDServing {
        namespace VectorSearch {

            class HashBasedDeduper
            {
            public:
                HashBasedDeduper();

                ~HashBasedDeduper();

                void Init(int stub = 0);

                void Clear();

                bool CheckAndSet(int idx);

            protected:
                // Max loop number in one hash block.
                static const int m_maxLoop = 8;

                // Max pool size.
                static const int m_poolSize = 8191;

                // Could we use the second hash block.
                bool m_secondHash;

                // Record 2 hash tables.
                // [0~m_poolSize + 1) is the first block.
                // [m_poolSize + 1, 2*(m_poolSize + 1)) is the second block;
                int m_hashTable[(m_poolSize + 1) * 2];

                inline unsigned hash_func2(int idx, int loop);

                inline unsigned hash_func(unsigned idx);

                inline int _CheckAndSet(int* hashTable, int idx);
            };
}
}
}