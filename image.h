// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#pragma once

#include "nn-types.h"

#include <QPixmap>
#include <QImage>

#include <string>
#include <array>

namespace Image {

float* readPngImageFile(const std::string &fileName, TensorShape &outShape);
void writePngImageFile(const float *pixels, const TensorShape &shape, const std::string &fileName);
float* readPixmap(const QPixmap &pixmap, TensorShape &outShape);
float* resizeImage(const float *pixels, const TensorShape &shapeOld, const TensorShape &shapeNew);
float* regionOfImage(const float *pixels, const TensorShape &shape, const std::array<unsigned,4> region);
QPixmap toQPixmap(const float *image, const TensorShape &shape);
void flipHorizontally(const TensorShape &shape, const float *imgSrc, float *imgDst);
void flipVertically(const TensorShape &shape, const float *imgSrc, float *imgDst);
void makeGrayscale(const TensorShape &shape, const float *imgSrc, float *imgDst);

}
