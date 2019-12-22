#pragma once

#include <vector>

// various data types involved in NN model data

enum OperatorType {
  NONE=0,
  AVERAGE_POOL_2D=1,
  CONV_2D,
  DEPTHWISE_CONV_2D,
  FULLY_CONNECTED,
  RESHAPE,
  SOFTMAX,
  SQUEEZE
};

class Tensor {
private:
  std::vector<unsigned> dims;
public:
  Tensor(unsigned dim1)
    {dims.push_back(dim1);}
  Tensor(unsigned dim1, unsigned dim2)
    {dims.push_back(dim1); dims.push_back(dim2);}
  Tensor(unsigned dim1, unsigned dim2, unsigned dim3)
    {dims.push_back(dim1); dims.push_back(dim2); dims.push_back(dim3);}
  Tensor(unsigned dim1, unsigned dim2, unsigned dim3, unsigned dim4)
    {dims.push_back(dim1); dims.push_back(dim2); dims.push_back(dim3); dims.push_back(dim4);}
  Tensor(unsigned dim1, unsigned dim2, unsigned dim3, unsigned dim4, unsigned dim5)
    {dims.push_back(dim1); dims.push_back(dim2); dims.push_back(dim3); dims.push_back(dim4); dims.push_back(dim5);}
  Tensor(unsigned dim1, unsigned dim2, unsigned dim3, unsigned dim4, unsigned dim5, unsigned dim6)
    {dims.push_back(dim1); dims.push_back(dim2); dims.push_back(dim3); dims.push_back(dim4); dims.push_back(dim5); dims.push_back(dim6);}
};

