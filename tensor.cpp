// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#include "tensor.h"

#include <limits>
#include <memory>

#include <assert.h>

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

float* computeArgMax(const TensorShape &inputShape, const float *input, const std::vector<float> &palette) {
	assert(palette.size()%3 == 0); // it's a color palette with {R,G,B}
	assert(*inputShape.rbegin() == palette.size()/3); // match the number of colors

	auto inputSize = flatSize(inputShape);
	auto nchannels = *inputShape.rbegin();
	std::unique_ptr<float> output(new float[inputSize/nchannels*3]);
	float *o = output.get();

	for (auto inpute = input+inputSize; input<inpute; ) {
		auto nc = nchannels;
		unsigned c = 0;
		float val = std::numeric_limits<float>::lowest();
		int maxChannel = -1;
		do {
			if (*input > val) {
				val = *input;
				maxChannel = c;
			}
			input++;
			c++;
		} while (--nc > 0);

		const float *p = &palette[maxChannel*3];
		*o++ = *p++;
		*o++ = *p++;
		*o++ = *p;
	}

	return output.release();
}

}
