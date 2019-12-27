#pragma once

#include "nn-types.h"

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

}
