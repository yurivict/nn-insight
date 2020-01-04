// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#pragma once

#include "tensor.h"

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

void Add(
	const TensorShape &input1Shape, const float *input1Data,
	const TensorShape &input2Shape, const float *input2Data,
	const TensorShape &outputShape, float *outputData
);

void Mul(
	const TensorShape &input1Shape, const float *input1Data,
	const TensorShape &input2Shape, const float *input2Data,
	const TensorShape &outputShape, float *outputData
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

}
