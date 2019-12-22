#pragma once

#include <vector>

// various data types involved in NN model data

typedef std::vector<unsigned> TensorShape;

size_t tensorFlatSize(const TensorShape &shape);
