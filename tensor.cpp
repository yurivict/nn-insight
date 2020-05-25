// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#include "misc.h"
#include "tensor.h"

#include <functional>
#include <fstream>
#include <limits>
#include <memory>
#include <streambuf>
#include <string>

#include <assert.h>
#include <nlohmann/json.hpp>

namespace Tensor {

size_t flatSize(const TensorShape &shape) {
	size_t sz = 1;
	for (auto d : shape)
		sz *= d;
	return sz;
}

size_t sizeBetweenDims(const TensorShape &shape, int dim1, int dim2) {
	size_t sz = 1;
	for (int d = dim1; d <= dim2; d++)
		sz *= shape[d];
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

bool canBeAnImage(const TensorShape &shape) {
	return shape.size() == 3/*HWC*/ && (shape[2]==3 || shape[2]==1);
}

void saveTensorDataAsJson(const TensorShape &shape, const float *data, const char *fileName) {
	using json = nlohmann::json;

	std::function<json(unsigned,unsigned,const float*)> one;
	one = [&shape,&one](unsigned level, unsigned step, const float *data) {
		if (level < shape.size()) {
			json j = json::array();
			auto sz = shape[level];
			step /= sz;
			for (unsigned i = 0, ie = sz; i < ie; i++, data += step)
				j.push_back(one(level+1, step, data));
			return j;
		} else
			return json(*data);
	};

	// convert to json
	json j = one(0, flatSize(shape), data);

	// save
	std::ofstream f;
	f.open(fileName, std::ios_base::out|std::ios_base::trunc);
	if (!f.good()) {
		PRINT_ERR("failed to open the json file for writing")
		return;
	}
	f << j;
	f.close();
}

bool readTensorDataAsJson(const char *fileName, const TensorShape &shape, std::shared_ptr<const float> &tensorData) {
	using json = nlohmann::json;

	std::ifstream f(fileName);
	if (!f)
		return false;

	std::string str((std::istreambuf_iterator<char>(f)),
	                 std::istreambuf_iterator<char>());

	auto shapeSize = flatSize(shape);
	std::unique_ptr<float> data(new float[shapeSize]);
	float *p = data.get(), *pe = p + shapeSize;

	std::function<bool(const json &j)> one;
	one = [&p,pe,&one](const json &j) {
		if (j.is_array()) {
			for (auto &e : j)
				if (!one(e))
					return false; // failed to import data
			return true;
		} else if (j.is_number()) {
			if (p == pe)
				return false; // no space for the next element: must be a shape mismatch
			*p++ = j.get<float>();
			return true;
		} else
			return false; // can't be any other JSON object type in the tensor data file
	};

	if (one(json::parse(str)) && p == pe) {
		tensorData.reset(data.release());
		return true;
	}

	return false;

}

}
