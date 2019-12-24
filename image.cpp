
#include "image.h"

#include <png++/png.hpp>

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

}
