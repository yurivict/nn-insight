#pragma once

#include <vector>

#include <stddef.h> // for size_t

// various data types involved in NN model data

typedef std::vector<unsigned> TensorShape;

size_t tensorFlatSize(const TensorShape &shape);
