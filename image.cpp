
#include "image.h"
#include "nn-types.h"

#include <png++/png.hpp>
#include <avir.h>

#include <memory>

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

}
