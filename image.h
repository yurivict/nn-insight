#pragma once

#include "nn-types.h"

#include <QPixmap>
#include <QImage>

#include <string>

namespace Image {

float* readPngImageFile(const std::string &fileName, TensorShape &outShape);
float* readPixmap(const QPixmap &pixmap, TensorShape &outShape);
float* resizeImage(const float *pixels, const TensorShape &shapeOld, const TensorShape &shapeNew);
QPixmap toQPixmap(const float *image, const TensorShape &shape);
void flipHorizontally(const TensorShape &shape, const float *imgSrc, float *imgDst);
void flipVertically(const TensorShape &shape, const float *imgSrc, float *imgDst);
void makeGrayscale(const TensorShape &shape, const float *imgSrc, float *imgDst);

}
