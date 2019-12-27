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


std::tuple<unsigned,unsigned> computePaddingValues(unsigned stride, unsigned dilationRate, unsigned inSize, int filterSize, int outSize) {
	// based on ComputePaddingWithOffset from the TF Lite project in order to match the results (XXX is this correct for every other format?)
	unsigned effectiveFilterSize = (filterSize-1) * dilationRate + 1;
	unsigned totalPadding = ((outSize-1)*stride + effectiveFilterSize - inSize);
	totalPadding = totalPadding > 0 ? totalPadding : 0;
	return {totalPadding/2, totalPadding%2}; // returns {padding,offset}
} 

