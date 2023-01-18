#include "inc/Helper/KeyValueIO.h"
#include <atomic>

namespace SPTAG {
    namespace SPANN {
        // concurrently safe with RocksDBIO
        class PersistentBuffer
        {
        public:
            PersistentBuffer(std::string& fileName, std::shared_ptr<Helper::KeyValueIO> db) : db(db), _size(0)
            {
                LOG(Helper::LogLevel::LL_Info, "SPFresh: persistent buffer: %s\n", fileName.c_str());
                db->Initialize(fileName.c_str(), true, true);
            }

            ~PersistentBuffer() {}

            inline int GetNewAssignmentID() { return _size++; }

            inline void GetAssignment(int assignmentID, std::string* assignment) { db->Get(assignmentID, assignment); }

            inline int PutAssignment(std::string& assignment)
            {
                int assignmentID = GetNewAssignmentID();
                db->Put(assignmentID, assignment);
                return assignmentID;
            }

            inline int GetCurrentAssignmentID() { return _size; }

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