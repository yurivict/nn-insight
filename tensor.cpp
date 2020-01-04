// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#include "tensor.h"

namespace Tensor {

size_t flatSize(const TensorShape &shape) {
	size_t sz = 1;
	for (auto d : shape)
		sz *= d;
	return sz;
}

unsigned numMultiDims(const TensorShape &shape) {
	unsigned numMultiDims = 0;
	for (auto d : shape)
		if (d > 1)
			numMultiDims++;
	return numMultiDims;
}

TensorShape getLastDims(const TensorShape &shape, unsigned ndims) {
	TensorShape s = shape;
	while (s.size() > ndims)
		s.erase(s.begin());
	return s;
}

TensorShape stripLeadingOnes(const TensorShape &shape) {
	TensorShape s = shape;
	while (!s.empty() && s[0]==1)
		s.erase(s.begin());
	return s;
}

bool isSubset(const TensorShape &shapeLarge, const TensorShape &shapeSmall) {
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

}
