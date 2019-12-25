#pragma once

#include "nn-types.h"

namespace Image {

float* readPngImageFile(const std::string &fileName, TensorShape &outShape);
float* resizeImage(const float *pixels, const TensorShape &shapeOld, const TensorShape &shapeNew);

}
