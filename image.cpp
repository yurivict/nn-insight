
#include "image.h"
#include "nn-types.h"
#include "misc.h"
#include "util.h"

#include <png++/png.hpp>
#include <avir.h>

#include <QPixmap>
#include <QImage>

#include <memory>
#include <cstring>

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

float* readPixmap(const QPixmap &pixmap, TensorShape &outShape) {
	const QImage image = pixmap.toImage();

	std::unique_ptr<float> data(new float[image.width()*image.height()*3]);
	auto pi = image.bits();
	auto pf = data.get();
	switch (image.format()) {
	case QImage::Format_RGB32: // 0xffRRGGBB
	case QImage::Format_ARGB32: { // 0xAARRGGBB
		for (float *pfe = pf+image.width()*image.height()*3; pf < pfe; pi+=4, pf+=3) {
			pf[0] = pi[2];
			pf[1] = pi[1];
			pf[2] = pi[0];
		}
		break;
	} case QImage::Format_RGB888: { // 24-bit RGB format (8-8-8)
		for (float *pfe = pf+image.width()*image.height()*3; pf < pfe; )
			*pf++ = *pi++;
		break;
	} default: {
		// we could also easily handle QImage::Format_RGB666, QImage::Format_RGB555
		WARNING("unable to handle the pixmap format=" << image.format() <<
		        ": width=" << image.width() << " height=" << image.height() << " bpl=" << image.bytesPerLine())
		return nullptr;
	}}

	outShape = {(unsigned)image.height(), (unsigned)image.width(), 3};
	return data.release();
}

float* resizeImage(const float *pixels, const TensorShape &shapeOld, const TensorShape &shapeNew) {
	auto sz = tensorFlatSize(shapeNew);
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

QPixmap toQPixmap(const float *image, const TensorShape &shape) {
	return QPixmap::fromImage(QImage(
		std::unique_ptr<const uchar>(Util::convertArrayFloatToUInt8(image, tensorFlatSize(shape))).get(),
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
	for (auto imgSrce = imgSrc + tensorFlatSize(shape); imgSrc<imgSrce; imgSrc+=rowSize, imgDst+=rowSize)
		reverseArray(imgSrc, imgDst, rowSize, blockSize);
}

void flipVertically(const TensorShape &shape, const float *imgSrc, float *imgDst) {
	reverseArray(imgSrc, imgDst, tensorFlatSize(shape), shape[1]*shape[2]);
}

void makeGrayscale(const TensorShape &shape, const float *imgSrc, float *imgDst) {
	assert(shape[2]==3);
	auto convertColor = [](float R, float G, float B) {
		return (R+G+B)/3;
	};
	for (auto imgSrce = imgSrc + tensorFlatSize(shape); imgSrc<imgSrce; imgSrc+=3, imgDst+=3)
		imgDst[0] = imgDst[1] = imgDst[2] = convertColor(imgSrc[0], imgSrc[1], imgSrc[2]);
}

}
