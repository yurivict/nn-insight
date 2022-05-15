// Copyright (C) 2022 by Yuri Victorovich. All rights reserved.

#pragma once

#include "tensor.h"

#include <array>

namespace NnOperators {

void Conv2D(
	const TensorShape &inputShape, const float *inputData,
	const TensorShape &filterShape, const float *filterData,
	const TensorShape &biasShape, const float *biasData,
	const TensorShape &outputShape, float *outputData,
	unsigned paddingWidth, unsigned paddingHeight,
	unsigned strideWidth, unsigned strideHeight,
	unsigned dilationWidthFactor, unsigned dilationHeightFactor
);

void DepthwiseConv2D(
	const TensorShape &inputShape, const float *inputData,
	const TensorShape &filterShape, const float *filterData,
	const TensorShape &biasShape, const float *biasData,
	const TensorShape &outputShape, float *outputData,
	unsigned paddingWidth, unsigned paddingHeight,
	unsigned strideWidth, unsigned strideHeight,
	unsigned dilationWidthFactor, unsigned dilationHeightFactor,
	unsigned depthMultiplier
);

void FullyConnected(
	const TensorShape &inputShape, const float *inputData,
	const TensorShape &filterShape, const float *filterData,
	const TensorShape &biasShape, const float *biasData,
	const TensorShape &outputShape, float *outputData
);

void MaxPool(
	const TensorShape &inputShape, const float *inputData,
	const TensorShape &outputShape, float *outputData,
	unsigned paddingWidth, unsigned paddingHeight,
	unsigned strideWidth, unsigned strideHeight,
	unsigned filterWidth, unsigned filterHeight
);

void AveragePool(
	const TensorShape &inputShape, const float *inputData,
	const TensorShape &outputShape, float *outputData,
	unsigned paddingWidth, unsigned paddingHeight,
	unsigned strideWidth, unsigned strideHeight,
	unsigned filterWidth, unsigned filterHeight
);

void Softmax(
	const TensorShape &inputShape, const float *inputData,
	const TensorShape &outputShape, float *outputData,
	float beta
);

void ResizeBilinear(
	const TensorShape &inputShape, const float *inputData,
	const TensorShape &outputShape, float *outputData,
	bool alignCorners
);

void ResizeNearestNeighbor(
	const TensorShape &inputShape, const float *inputData,
	const TensorShape &outputShape, float *outputData,
	bool alignCorners
);

void LocalResponseNormalization(
	const TensorShape &inputShape, const float *inputData,
	const TensorShape &outputShape, float *outputData,
	int radius, float alpha, float beta, float bias
);

void Mean(
	const TensorShape &inputShape, const float *inputData,
	const TensorShape &outputShape, float *outputData,
	const int32_t *axis, unsigned axis_count
);

void Pad(
	const std::array<int32_t,2>* paddings,
	const TensorShape &inputShape, const float *inputData,
	const TensorShape &outputShape, float *outputData
);

}
