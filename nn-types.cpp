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

