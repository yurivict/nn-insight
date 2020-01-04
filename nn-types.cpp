// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#include "nn-types.h"

std::tuple<unsigned,unsigned> computePaddingValues(unsigned stride, unsigned dilationRate, unsigned inSize, int filterSize, int outSize) {
	// based on ComputePaddingWithOffset from the TF Lite project in order to match the results (XXX is this correct for every other format?)
	unsigned effectiveFilterSize = (filterSize-1) * dilationRate + 1;
	unsigned totalPadding = ((outSize-1)*stride + effectiveFilterSize - inSize);
	totalPadding = totalPadding > 0 ? totalPadding : 0;
	return {totalPadding/2, totalPadding%2}; // returns {padding,offset}
} 

