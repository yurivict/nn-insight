#pragma once

#include <vector>

#include <stddef.h> // for size_t

#include <vector>
#include <tuple>

// various data types involved in NN model data

typedef std::vector<unsigned> TensorShape;

enum WidthHeight {
	WIDTH,
	HEIGHT
};

enum InputNormalizationRange {
	InputNormalizationRange_0_1,
	InputNormalizationRange_0_255,
	InputNormalizationRange_0_128,
	InputNormalizationRange_0_64,
	InputNormalizationRange_0_32,
	InputNormalizationRange_0_16,
	InputNormalizationRange_0_8,
	InputNormalizationRange_M1_P1,
	InputNormalizationRange_ImageNet
};

enum InputNormalizationColorOrder {
	InputNormalizationColorOrder_RGB,
	InputNormalizationColorOrder_BGR
};

typedef std::tuple<InputNormalizationRange,InputNormalizationColorOrder> InputNormalization;

size_t tensorFlatSize(const TensorShape &shape);
unsigned tensorNumMultiDims(const TensorShape &shape);
TensorShape tensorGetLastDims(const TensorShape &shape, unsigned ndims);
