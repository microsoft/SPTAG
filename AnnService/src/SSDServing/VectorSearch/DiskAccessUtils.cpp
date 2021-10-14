#include "inc/SSDServing/VectorSearch/DiskAccessUtils.h"
#include <Windows.h>
#include <tchar.h>

namespace SPTAG {
    namespace SSDServing {
        namespace VectorSearch
        {
            namespace DiskUtils
            {
                uint64_t GetSectorSize(const char* p_filePath)
                {
                    DWORD dwSectorSize = 0;
                    LPTSTR lpFilePart;
                    // Get file sector size
                    DWORD dwSize = GetFullPathName(p_filePath, 0, NULL, &lpFilePart);
                    if (dwSize > 0)
                    {
                        LPTSTR lpBuffer = new TCHAR[dwSize];
                        if (lpBuffer)
                        {
                            DWORD dwResult = GetFullPathName(p_filePath, dwSize, lpBuffer, &lpFilePart);
                            if (dwResult > 0 && dwResult <= dwSize)
                            {
                                bool nameValid = false;
                                if (lpBuffer[0] == _T('\\'))
                                {
                                    if (lpBuffer[1] == _T('\\'))
                                    {
                                        DWORD i;
                                        if (dwSize > 2)
                                        {
                                            for (i = 2; lpBuffer[i] != 0 && lpBuffer[i] != _T('\\'); i++);
                                            if (lpBuffer[i] == _T('\\'))
                                            {
                                                for (i++; lpBuffer[i] != 0 && lpBuffer[i] != _T('\\'); i++);
                                                if (lpBuffer[i] == _T('\\'))
                                                {
                                                    lpBuffer[i + 1] = 0;
                                                    nameValid = true;
                                                }
                                            }
                                        }
                                    }
                                }
                                else
                                {
                                    if (((lpBuffer[0] >= _T('a') && lpBuffer[0] <= _T('z')) || (lpBuffer[0] >= _T('A') && lpBuffer[0] <= _T('Z'))) &&
                                        lpBuffer[1] == _T(':'))
                                    {
                                        if (lpBuffer[2] != 0)
                                        {
                                            lpBuffer[2] = _T('\\'); lpBuffer[3] = 0;
                                            nameValid = true;
                                        }
                                    }
                                }
                                if (nameValid)
                                {
                                    DWORD dwSPC, dwNOFC, dwTNOC;
                                    GetDiskFreeSpace(lpBuffer, &dwSPC, &dwSectorSize, &dwNOFC, &dwTNOC);
                                }
                            }
                            delete[] lpBuffer;
                        }
                    }

                    return dwSectorSize;
                }
            }
        }
    }
}