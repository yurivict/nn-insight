// Copyright (C) 2022 by Yuri Victorovich. All rights reserved.

#include "nn-operators.h"
#include "misc.h"
#include "tensor.h"

#include <array>
#include <cstring>

namespace NnOperators {
/*
void MirrorPad(
	const std::array<int32_t,2>* paddings,
	const TensorShape &inputShape, const float *inputData,
	const TensorShape &outputShape, float *outputData
) {
	// checks
	if (inputShape.size() != outputShape.size())
		FAIL("MirrorPad: input and output shape mismatch")
	for (unsigned i = 0, ie = inputShape.size(); i < ie; i++)
		if (inputShape[i] + paddings[i][0] + paddings[i][1] != outputShape[i])
			FAIL("MirrorPad: input and output shape mismatch")

	// find effective dimensions/shape
	std::array<int32_t,2> effectivePaddings[inputShape.size()];
	TensorShape effectiveShape;
	unsigned numEffectiveDims = 0;
	{
		std::array<int32_t,2> *ep = effectivePaddings;
		for (unsigned i = 0, ie = inputShape.size(); i < ie; i++)
			if (i==0 || paddings[i][0]+paddings[i][1] > 0) {
				*ep++ = paddings[i];
				effectiveShape.push_back(inputShape[i]);
				numEffectiveDims++;
			} else {
				*effectiveShape.rbegin() *= inputShape[i];
			}
	}

	std::memset(outputData, 0, Tensor::flatSize(effectiveShape)*sizeof(float));

	// dimension sizes
	unsigned sizes[numEffectiveDims];
	{
		unsigned sz = 1;
		for (int i = numEffectiveDims-1; i >= 0; i--) {
			sizes[i] = sz;
			sz *= effectiveShape[i];
		}
	}

	PRINT("numEffectiveDims=" << numEffectiveDims)
	for (auto sz : sizes)
		PRINT("... sz=" << sz)
	for (auto D : effectiveShape)
		PRINT("... shape/D=" << D)

	FAIL("for now")
}
*/
}
