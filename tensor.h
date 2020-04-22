// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#include <vector>

typedef std::vector<unsigned> TensorShape;

namespace Tensor {

size_t flatSize(const TensorShape &shape);
unsigned numMultiDims(const TensorShape &shape);
TensorShape getLastDims(const TensorShape &shape, unsigned ndims);
TensorShape stripLeadingOnes(const TensorShape &shape);
bool isSubset(const TensorShape &shapeLarge, const TensorShape &shapeSmall);
float* computeArgMax(const TensorShape &inputShape, const float *input, const std::vector<float> &palette);
bool canBeAnImage(const TensorShape &shape);

}
