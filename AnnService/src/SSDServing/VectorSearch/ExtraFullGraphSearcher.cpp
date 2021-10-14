#include "inc/SSDServing/VectorSearch/ExtraFullGraphSearcher.h"

void SPTAG::SSDServing::VectorSearch::ErrorExit() {
	// Retrieve the system error message for the last-error code

	LPVOID lpMsgBuf;
	DWORD dw = GetLastError();

	FormatMessage(
		FORMAT_MESSAGE_ALLOCATE_BUFFER |
		FORMAT_MESSAGE_FROM_SYSTEM |
		FORMAT_MESSAGE_IGNORE_INSERTS,
		NULL,
		dw,
		0,
		(LPTSTR)&lpMsgBuf,
		0, NULL);

	// Display the error message and exit the process

	LOG(Helper::LogLevel::LL_Error, "Failed with: %s\n", (char*)lpMsgBuf);

	LocalFree(lpMsgBuf);
	ExitProcess(dw);
}
