#pragma once

#include "nn-types.h"

#include <QPixmap>

#include <string>

namespace Image {

float* readPngImageFile(const std::string &fileName, TensorShape &outShape);
float* readPixmap(const QPixmap &pixmap, TensorShape &outShape);
float* resizeImage(const float *pixels, const TensorShape &shapeOld, const TensorShape &shapeNew);

}
