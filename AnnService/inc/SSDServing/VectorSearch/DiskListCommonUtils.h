#pragma once

#include <memory>

#include <Windows.h>

namespace SPTAG {
    namespace SSDServing {
        namespace VectorSearch
        {
            class HandleWrapper
            {
            public:
                HandleWrapper(HANDLE p_handle) : m_handle(p_handle) {}
                HandleWrapper(HandleWrapper&& p_right) : m_handle(std::move(p_right.m_handle)) {}
                HandleWrapper() : m_handle(INVALID_HANDLE_VALUE) {}
                ~HandleWrapper() {}

                void Reset(HANDLE p_value) { m_handle.reset(p_value); }
                HANDLE GetHandle() { return m_handle.get(); }
                bool IsValid() const { return m_handle.get() != INVALID_HANDLE_VALUE; };
                void Close() { m_handle.reset(INVALID_HANDLE_VALUE); }

                HandleWrapper(const HandleWrapper&) = delete;
                HandleWrapper& operator=(const HandleWrapper&) = delete;

                struct HandleDeleter
                {
                    void operator()(HANDLE p_handle) const
                    {
                        if (p_handle != INVALID_HANDLE_VALUE)
                        {
                            ::CloseHandle(p_handle);
                        }

                        p_handle = INVALID_HANDLE_VALUE;
                    }
                };

            private:
                typedef std::unique_ptr<typename std::remove_pointer<HANDLE>::type, HandleDeleter> UniqueHandle;
                UniqueHandle m_handle;
            };
        }
    }
}