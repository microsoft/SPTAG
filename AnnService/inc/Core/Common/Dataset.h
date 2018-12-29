#ifndef _SPTAG_COMMON_DATASET_H_
#define _SPTAG_COMMON_DATASET_H_

#include <fstream>

#define ALIGN 32

#define aligned_malloc(a, b) _mm_malloc(a, b)
#define aligned_free(a) _mm_free(a)

namespace SPTAG
{
    namespace COMMON
    {
        // structure to save Data and Graph
        template <typename T>
        class Dataset
        {
        private:
            int rows;
            int cols;
            bool ownData = false;
            T* data = nullptr;
            std::vector<T>* dataIncremental = nullptr;

        public:
            Dataset() {}
            Dataset(int rows_, int cols_, T* data_ = nullptr)
            {
                Initialize(rows_, cols_, data_);
            }
            ~Dataset()
            {
                if (ownData) aligned_free(data);
                if (dataIncremental) {
                    dataIncremental->clear();
                    delete dataIncremental;
                }
            }
            void Initialize(int rows_, int cols_, T* data_ = nullptr)
            {
                rows = rows_;
                cols = cols_;
                data = data_;
                if (data == nullptr)
                {
                    ownData = true;
                    data = (T*)aligned_malloc(sizeof(T) * rows * cols, ALIGN);
                }
                dataIncremental = new std::vector<T>();
            }
            void SetR(int R_) 
            {
                if (R_ >= rows)
                    dataIncremental->resize((R_ - rows) * cols);
                else 
                {
                    rows = R_;
                    dataIncremental->clear();
                }
            }
            int R() const { return (int)(rows + dataIncremental->size() / cols); }
            int C() const { return cols; }
            T* operator[](int index)
            {
                if (index >= rows) {
                    return dataIncremental->data() + (size_t)(index - rows)*cols;
                }
                return data + (size_t)index*cols;
            }
            const T* operator[](int index) const
            {
                if (index >= rows) {
                    return dataIncremental->data() + (size_t)(index - rows)*cols;
                }
                return data + (size_t)index*cols;
            }

            T* GetData()
            {
                return data;
            }

            void reset() 
            {
                if (ownData) {
                    aligned_free(data);
                    ownData = false;
                }
                if (dataIncremental) {
                    dataIncremental->clear();
                    delete dataIncremental;
                }
            }
            
            void AddBatch(const T* pData, int num)
            {
                dataIncremental->insert(dataIncremental->end(), pData, pData + num*cols);
            }

            void AddBatch(int num)
            {
                dataIncremental->insert(dataIncremental->end(), (size_t)num*cols, T(-1));
            }
        };
    }
}

#endif // _SPTAG_COMMON_DATASET_H_
