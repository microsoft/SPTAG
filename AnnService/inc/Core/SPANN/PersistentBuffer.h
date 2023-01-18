#include "inc/Helper/KeyValueIO.h"
#include <atomic>

namespace SPTAG {
    namespace SPANN {
        // concurrently safe with RocksDBIO
        class PersistentBuffer
        {
        public:
            PersistentBuffer(std::shared_ptr<Helper::KeyValueIO> db) : db(db), _size(0) { }

            ~PersistentBuffer() {}

            inline int GetNewAssignmentID() { return _size++; }

            inline int PutAssignment(std::string& assignment)
            {
                int assignmentID = GetNewAssignmentID();
                db->Put(assignmentID, assignment);
                return assignmentID;
            }

            inline bool StartToScan(std::string& assignment)
            {
                SizeType newSize = 0;
                if (db->StartToScan(newSize, &assignment) != ErrorCode::Success) return false;
                _size = newSize+1;
                return true;
            }

            inline bool NextToScan(std::string& assignment)
            {
                SizeType newSize = 0;
                if (db->NextToScan(newSize, &assignment) != ErrorCode::Success) return false;
                _size = newSize+1;
                return true;
            }

            inline void ClearPreviousRecord() 
            {
                db->DeleteRange(0, _size.load());
                _size = 0;
            }

            inline int StopPB()
            {
                db->ShutDown();
                return 0;
            }

        private:
            std::shared_ptr<Helper::KeyValueIO> db;
            std::atomic_int _size;
        };
    }
}