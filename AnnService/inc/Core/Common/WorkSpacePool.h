// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_WORKSPACEPOOL_H_
#define _SPTAG_COMMON_WORKSPACEPOOL_H_

#include "WorkSpace.h"
#include "inc/Helper/ConcurrentSet.h"

#include <list>
#include <mutex>
#include <stdarg.h>

namespace SPTAG
{
    namespace COMMON
    {

        template<typename T>
        class WorkSpacePool
        {
        public:
            WorkSpacePool() {}

            ~WorkSpacePool() 
            {
                std::shared_ptr<T> workspace;
                while (m_workSpacePool.try_pop(workspace))
                {
                    workspace.reset();
                }
                T::Reset();
            }

            std::shared_ptr<T> Rent()
            {
                std::shared_ptr<T> workSpace;
                {
                    if (m_workSpacePool.try_pop(workSpace))
                    {
                    }
                    else
                    {
                        workSpace.reset(new T(m_workSpace));
                    }
                }
                return workSpace;
            }

            void Return(const std::shared_ptr<T>& p_workSpace)
            {
                m_workSpacePool.push(p_workSpace);
            }

            void Init(int size, ...)
            {
                va_list args;
                va_start(args, size);
                m_workSpace.Initialize(args);
                va_end(args);
                for (int i = 0; i < size; i++)
                {
                    std::shared_ptr<T> workSpace(new T(m_workSpace));
                    m_workSpacePool.push(std::move(workSpace));
                }
            }

        private:
            Helper::Concurrent::ConcurrentQueue<std::shared_ptr<T>> m_workSpacePool;
            T m_workSpace;
        };

    }
}

#endif // _SPTAG_COMMON_WORKSPACEPOOL_H_
