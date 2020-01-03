// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

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
	InputNormalizationRange_M05_P05,
	InputNormalizationRange_14_34,
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
bool tensorIsSubset(const TensorShape &shapeLarge, const TensorShape &shapeSmall);

// based on ComputePaddingWithOffset from the TF Lite project in order to match the results
std::tuple<unsigned,unsigned> computePaddingValues(unsigned stride, unsigned dilationRate, unsigned inSize, int filterSize, int outSize);
