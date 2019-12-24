#pragma once

#include "nn-types.h"

namespace Image {

float* readPngImageFile(const std::string &fileName, TensorShape &outShape);

}
