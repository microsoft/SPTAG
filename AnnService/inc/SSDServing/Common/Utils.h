#include "inc/SSDServing/Common/stdafx.h"

template<typename T>
float Std(T* nums, int size) {
	float var = 0;

	float mean = 0;
	for (size_t i = 0; i < size; i++)
	{
		mean += nums[i] / static_cast<float>(size);
	}

	for (size_t i = 0; i < size; i++)
	{
		var += (nums[i] - mean) * (nums[i] - mean);
	}
	var /= static_cast<float>(size);

	return sqrt(var);
}