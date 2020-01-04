// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#include "image.h"
#include "tensor.h"
#include "misc.h"
#include "util.h"

#include <png++/png.hpp>
#include <avir/avir.h>

#include <QPixmap>
#include <QImage>

#include <string>
#include <array>
#include <memory>
#include <cstring>
#include <functional>

#include <assert.h>

namespace Image {

float* readPngImageFile(const std::string &fileName, TensorShape &outShape) {
	png::image<png::rgb_pixel> image(fileName.c_str());
	auto width  = image.get_width();
	auto height = image.get_height();

	std::unique_ptr<float> data(new float[width*height*3]);

	float *p = data.get();
	for (unsigned y = 0; y < height; ++y)
		for (unsigned x = 0; x < width; ++x) {
			png::rgb_pixel color = image[y][x];
			for (auto c : {color.red, color.green, color.blue})
				*p++ = c;
		}

	outShape = {height,width,3};
	return data.release();
}

void writePngImageFile(const float *pixels, const TensorShape &shape, const std::string &fileName) { // ASSUME 0..255 normalization
	assert(shape.size()==3);
	auto width = shape[1];
	auto height = shape[0];

	png::image<png::rgb_pixel> image(width, height);
	for (unsigned y = 0; y < height; y++)
		for (unsigned x = 0; x < width; x++) {
			png::rgb_pixel &color = image[y][x];
			color.red   = *pixels++;
			color.green = *pixels++;
			color.blue  = *pixels++;
		}

	image.write(fileName);
}

float* readPixmap(const QPixmap &pixmap, TensorShape &outShape, std::function<void(const std::string&)> cbWarningMessage) {
	QImage image = pixmap.toImage();

	// always convert image to RGB32 so we don't have to deal with any other formats
#if QT_VERSION >= QT_VERSION_CHECK(5,13,0) // QImage::convertTo exists since Qt-5.13
	image.convertTo(QImage::Format_RGB32, Qt::ColorOnly); // CAVEAT: the alpha channel is converted to black, which isn't normally desirable
	assert(image.format()==QImage::Format_RGB32);
#else
	if (image.format()!=QImage::Format_RGB32) {
		cbWarningMessage(STR("Your Qt version " << QT_VERSION_MAJOR << "." << QT_VERSION_MINOR << "." << QT_VERSION_PATCH << " is older than 5.13.0, it is missing QImage::convertTo needed to convert this image to the QImage::Format_RGB32 format"));
		return nullptr;
	}
#endif

	// TODO for pixmaps with alpha-channel use QImage::Format_ARGB32 and convert alpha to white (or any other background color)
	if (pixmap.hasAlpha())
		WARNING("the image has alpha channel which is converted to black")

	std::unique_ptr<float> data(new float[image.width()*image.height()*3]);
	auto pi = image.bits();
	auto pf = data.get();

	for (float *pfe = pf+image.width()*image.height()*3; pf < pfe; pi+=4, pf+=3) {
		pf[0] = pi[2];
		pf[1] = pi[1];
		pf[2] = pi[0];
	}

	outShape = {(unsigned)image.height(), (unsigned)image.width(), 3};
	return data.release();
}

float* resizeImage(const float *pixels, const TensorShape &shapeOld, const TensorShape &shapeNew) {
	auto sz = Tensor::flatSize(shapeNew);
	std::unique_ptr<float> pixelsNew(new float[sz]);
	avir::CImageResizer<> ImageResizer(8);
	ImageResizer.resizeImage(
		pixels,
		shapeOld[1],
		shapeOld[0],
		0,
		pixelsNew.get(),
		shapeNew[1],
		shapeNew[0],
		shapeNew[2],
		0);

	// clip values because the resizer leaves some slightly out-of-range (0..255) values
	for (auto d = pixelsNew.get(), de = d + sz; d < de; d++) {
		if (*d < 0.)
			*d = 0.;
		else if (*d >= 255.)
			*d = 255.; // - std::numeric_limits<float>::epsilon();
	}

	return pixelsNew.release();
}

float* regionOfImage(const float *pixels, const TensorShape &shape, const std::array<unsigned,4> region) {
	assert(shape.size()==3);
	assert(region[0]<=region[2] && region[2]<shape[1]); // W
	assert(region[1]<=region[3] && region[3]<shape[0]); // H

	unsigned NC = shape[2];

	unsigned regionWidth  = region[2]-region[0]+1;
	unsigned regionHeight = region[3]-region[1]+1;
	std::unique_ptr<float> result(new float[regionHeight*regionWidth*NC]);

	unsigned skip = shape[1]*NC;
	unsigned bpl = regionWidth*NC;
	auto src = pixels+(region[1]*shape[1]+region[0])*NC;
	auto dst = result.get();
	for (auto dste = dst+regionWidth*regionHeight*NC; dst<dste; src+=skip, dst+=bpl)
		std::memcpy(dst, src, bpl*sizeof(float));

	return result.release();
}

QPixmap toQPixmap(const float *image, const TensorShape &shape) {
	return QPixmap::fromImage(QImage(
		std::unique_ptr<const uchar>(Util::convertArrayFloatToUInt8(image, Tensor::flatSize(shape))).get(),
		shape[1], // width
		shape[0], // height
		shape[1]*shape[2], // bytesPerLine
		QImage::Format_RGB888
	));
}

template<typename T>
static void reverseArray(const T *src, T *dst, unsigned rowSize, unsigned blockSize) {
	dst = dst + rowSize-blockSize;
	for (auto srce = src + rowSize; src<srce; src+=blockSize, dst-=blockSize)
		std::memcpy(dst, src, blockSize*sizeof(T));
}

void flipHorizontally(const TensorShape &shape, const float *imgSrc, float *imgDst) {
	unsigned rowSize = shape[1]*shape[2];
	unsigned blockSize = shape[2];
	for (auto imgSrce = imgSrc + Tensor::flatSize(shape); imgSrc<imgSrce; imgSrc+=rowSize, imgDst+=rowSize)
		reverseArray(imgSrc, imgDst, rowSize, blockSize);
}

void flipVertically(const TensorShape &shape, const float *imgSrc, float *imgDst) {
	reverseArray(imgSrc, imgDst, Tensor::flatSize(shape), shape[1]*shape[2]);
}

void makeGrayscale(const TensorShape &shape, const float *imgSrc, float *imgDst) {
	assert(shape[2]==3);
	auto convertColor = [](float R, float G, float B) {
		return (R+G+B)/3;
	};
	for (auto imgSrce = imgSrc + Tensor::flatSize(shape); imgSrc<imgSrce; imgSrc+=3, imgDst+=3)
		imgDst[0] = imgDst[1] = imgDst[2] = convertColor(imgSrc[0], imgSrc[1], imgSrc[2]);
}

}
