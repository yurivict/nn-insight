// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#include "nn-types.h"

size_t tensorFlatSize(const TensorShape &shape) {
	size_t sz = 1;
	for (auto d : shape)
		sz *= d;
	return sz;
}

unsigned tensorNumMultiDims(const TensorShape &shape) {
	unsigned numMultiDims = 0;
	for (auto d : shape)
		if (d > 1)
			numMultiDims++;
	return numMultiDims;
}

TensorShape tensorGetLastDims(const TensorShape &shape, unsigned ndims) {
	TensorShape s = shape;
	while (s.size() > ndims)
		s.erase(s.begin());
	return s;
}

TensorShape tensorStripLeadingOnes(const TensorShape &shape) {
	TensorShape s = shape;
	while (!s.empty() && s[0]==1)
		s.erase(s.begin());
	return s;
}

bool tensorIsSubset(const TensorShape &shapeLarge, const TensorShape &shapeSmall) {
	// check size
	if (shapeLarge.size() < shapeSmall.size())
		return false;

	// strip ones
	TensorShape small = shapeSmall;
	while (!small.empty() && small[0]==1)
		small.erase(small.begin());
	
	// compare
	for (TensorShape::const_reverse_iterator it = small.rbegin(), ite = small.rend(), itl = shapeLarge.rbegin(); it!=ite; it++, itl++)
		if (*it != *itl)
			return false;

	// all matched
	return true;
}

std::tuple<unsigned,unsigned> computePaddingValues(unsigned stride, unsigned dilationRate, unsigned inSize, int filterSize, int outSize) {
	// based on ComputePaddingWithOffset from the TF Lite project in order to match the results (XXX is this correct for every other format?)
	unsigned effectiveFilterSize = (filterSize-1) * dilationRate + 1;
	unsigned totalPadding = ((outSize-1)*stride + effectiveFilterSize - inSize);
	totalPadding = totalPadding > 0 ? totalPadding : 0;
	return {totalPadding/2, totalPadding%2}; // returns {padding,offset}
} 

