#ifndef _SPTAG_COMMON_LOG_H_
#define _SPTAG_COMMON_LOG_H_

#include "inc/Core/VectorIndex.h"

/**
 * A Write Ahead Log (WAL) provides service for reading, writing waledits.
 * 
**/
namespace SPTAG
{
    namespace COMMON
    {
        /*
        template <typename T>
        struct IndexDataSet
        {
        public:
            IndexDataSet(T* p_data, SizeType p_length): data(p_data), length(p_length) {}

            void SaveIndexDataSet(std::shared_ptr<Helper::DiskIO> out) 
            {
                IOBINARY(out, WriteBinary, sizeof(SizeType), (const char*)&length);
                IOBINARY(out, WriteBinary, length * sizeof(T), reinterpret_cast<const char*>(data));
            }

            void Load(std::shared_ptr<Helper::DiskIO> in)
            {
                IOBINARY(in, ReadBinary, sizeof(SizeType), (char*)&length);
                T* data = new T[length];
                IOBINARY(in, ReadBinary, length *sizeof(T), (char*)&data);
            }

            void Destroy()
            {
                delete[] data;
            }

            T* data;
            SizeType length;
        };
        */

        class WAL
        {
        public:
            virtual ~WAL();

            virtual void Init(const char* path) = 0;

            virtual ErrorCode Start() = 0;

            virtual bool Append(VectorSet* indexdataSet, MetadataSet* metadataSet) = 0;

            virtual ErrorCode Replay(VectorIndex* index) = 0;

        };

        class SimpleFileWAL: public WAL
        {
        public:

            ~SimpleFileWAL() {}

            ErrorCode Init(const std::string path);

            ErrorCode Start();

            bool Append(VectorSet* indexdataSet, MetadataSet* metadataSet);

            ErrorCode Replay(VectorIndex* index);
        private:
            static const std：string metaFile = "meta";
            static const std：string metaindexFile = "metaindex";
            static const std：string indexFile = "data";

            const std：string dir;

            std::shared_ptr<Helper::DiskIO> metaIO;
            std::shared_ptr<Helper::DiskIO> metaindexIO;
            std::shared_ptr<Helper::DiskIO> indexIO;

        };
    } // namespace COMMON
} // namespace SPTAG

#endif // _SPTAG_COMMON_LOG_H_