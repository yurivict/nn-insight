// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#include <vector>

typedef std::vector<unsigned> TensorShape;

namespace Tensor {

size_t flatSize(const TensorShape &shape);
size_t sizeBetweenDims(const TensorShape &shape, int dim1, int dim2); // between [dim1 .. dim2], therefore they should be 'int'
unsigned numMultiDims(const TensorShape &shape);
TensorShape getLastDims(const TensorShape &shape, unsigned ndims);
TensorShape stripLeadingOnes(const TensorShape &shape);
bool isSubset(const TensorShape &shapeLarge, const TensorShape &shapeSmall);
float* computeArgMax(const TensorShape &inputShape, const float *input, const std::vector<float> &palette);
bool canBeAnImage(const TensorShape &shape);
void saveTensorDataAsJson(const TensorShape &shape, const float *data, const char *fileName);
bool readTensorDataAsJson(const char *fileName, const TensorShape &shape, std::shared_ptr<const float> &tensorData);

}
