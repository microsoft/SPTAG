#ifndef _SPTAG_BKT_DATASET_H_
#define _SPTAG_BKT_DATASET_H_

#include <fstream>

#define ALIGN 32

#define aligned_malloc(a, b) _mm_malloc(a, b)
#define aligned_free(a) _mm_free(a)

namespace SPTAG
{
    namespace BKT 
    {
        template <typename T>
        class LRUCache
        {
        private:
            struct Item
            {
                int idx;
                T* data;
                Item* next;
                Item(int idx_, T* data_) : idx(idx_), data(data_), next(nullptr) {}
            };

            int rows;
            int cols;
            T* data;
            std::ifstream fp;
            std::unordered_map<int, Item*> cache;
            Item* head;

        public:
            LRUCache(const char * filename_, int caches_ = 10000000)
            {
                fp.open(filename_, std::ifstream::binary);
                fp.read((char *)&rows, sizeof(int));
                fp.read((char *)&cols, sizeof(int));
                if (caches_ > rows) caches_ = rows;

                data = (T*)aligned_malloc(sizeof(T) * caches_ * cols, ALIGN);

                int i = 0, batch = 10000;
                while (i + batch < caches_)
                {
                    fp.read((char *)(data + (size_t)i*cols), sizeof(T)*cols*batch);
                    i += batch;
                }
                fp.read((char *)(data + (size_t)i*cols), sizeof(T)*cols*(caches_ - i));

                Item *p = head = new Item(-1, nullptr);
                for (i = 0; i < caches_; i++)
                {
                    p->next = cache[(i + 1) % caches_] = new Item(i, data + (size_t)i*cols);
                    p = p->next;
                }
                p->next = head->next;
                delete head;
                head = p;

                std::cout << "Use LRUCache (" << caches_ << ")" << std::endl;
            }
            ~LRUCache()
            {
                fp.close();

                Item *p = head->next, *q;
                while (p != head)
                {
                    q = p;
                    p = p->next;
                    delete q;
                }
                delete head;

                aligned_free(data);
            }
            int R() { return rows; }
            int C() { return cols; }
            T* get(int index)
            {
                auto iter = cache.find(index);
                if (iter == cache.end())
                {
                    Item *p = head->next;
                    cache[index] = cache[p->idx];
                    cache.erase(p->idx);
                    p->idx = index;
                    fp.seekg(sizeof(int) + sizeof(int) + index * sizeof(T) * cols, std::ios_base::beg);
                    fp.read((char *)p->data, sizeof(T)*cols);
                    head = p;
                    return p->data;
                }
                else
                {
                    Item *p = iter->second, *q = p->next;
                    if (q == head || q == head->next)
                    {
                        head = q;
                        return q->data;
                    }
                    p->next = q->next;
                    cache[q->next->idx] = p;

                    q->next = head->next;
                    head->next = q;
                    cache[index] = head;
                    cache[q->next->idx] = q;

                    head = q;
                    return q->data;
                }
            }
        };

        // structure to save Data and Graph
        template <typename T>
        class Dataset
        {
        private:
            int rows;
            int cols;
            bool ownData = false;
            T* data = nullptr;
            LRUCache<T>* cache = nullptr;
            std::vector<T>* dataIncremental = nullptr;

        public:
            Dataset() {}
            Dataset(int rows_, int cols_, T* data_ = nullptr, const char * filename_ = nullptr, int cachesize_ = 0) { Initialize(rows_, cols_, data_, filename_, cachesize_); }
            ~Dataset()
            {
                if (cache != nullptr) delete cache;
                if (ownData) aligned_free(data);
                if (dataIncremental) {
                    dataIncremental->clear();
                    delete dataIncremental;
                }
            }
            void Initialize(int rows_, int cols_, T* data_ = nullptr, const char * filename_ = nullptr, int cachesize_ = 0)
            {
                if (filename_ != nullptr)
                {
                    cache = new LRUCache<T>(filename_, cachesize_);
                    rows = cache->R();
                    cols = cache->C();
                }
                else
                {
                    rows = rows_;
                    cols = cols_;
                    data = data_;
                    if (data == nullptr)
                    {
                        ownData = true;
                        data = (T*)aligned_malloc(sizeof(T) * rows * cols, ALIGN);
                    }
                }
                dataIncremental = new std::vector<T>();
            }
            void SetR(int R_) { rows = R_; }
            int R() const { return (int)(rows + dataIncremental->size() / cols); }
            int C() const { return cols; }
            T* operator[](int index)
            {
                if (index >= rows) {
                    return dataIncremental->data() + (size_t)(index - rows)*cols;
                }
                if (cache != nullptr)
                {
                    return cache->get(index);
                }
                else
                {
                    return data + (size_t)index*cols;
                }
            }
            const T* operator[](int index) const
            {
                if (index >= rows) {
                    return dataIncremental->data() + (size_t)(index - rows)*cols;
                }
                if (cache != nullptr)
                {
                    return cache->get(index);
                }
                else
                {
                    return data + (size_t)index*cols;
                }
            }
            T* GetData() {
                return data;
            }

            void AddBatch(const T* pData, int num)
            {
                dataIncremental->insert(dataIncremental->end(), pData, pData + num*cols);
            }

            void AddReserved(int num)
            {
                dataIncremental->insert(dataIncremental->end(), (size_t)num*cols, T(-1));
            }
        };
    }
}

#endif // _SPTAG_BKT_DATASET_H_
