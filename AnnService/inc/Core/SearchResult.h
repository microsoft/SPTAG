#ifndef _SPTAG_SEARCHRESULT_H_
#define _SPTAG_SEARCHRESULT_H_

#include "CommonDataStructure.h"

namespace SPTAG
{
	struct BasicResult
	{
		int VID;
		float Dist;
		ByteArray Meta;

		BasicResult() : VID(-1), Dist(MaxDist) {}

		BasicResult(int p_vid, float p_dist) : VID(p_vid), Dist(p_dist) {}

		BasicResult(int p_vid, float p_dist, ByteArray p_meta) : VID(p_vid), Dist(p_dist), Meta(p_meta) {}
	};

} // namespace SPTAG

#endif // _SPTAG_SEARCHRESULT_H_