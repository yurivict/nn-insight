
#include "compute.h"
#include "plugin-interface.h"
#include "nn-types.h"
#include "image.h"
#include "misc.h"
#include "util.h"

#include <string>
#include <vector>
#include <memory>

namespace Compute {

bool compute(
	const PluginInterface::Model *model,
	std::shared_ptr<float> &inputTensor, const TensorShape &inputShape,
	std::unique_ptr<std::vector<std::shared_ptr<const float>>> &tensorData,
	std::string &outWarningMessage)
{

	// allocate tensors array
	if (!tensorData) {
		tensorData.reset(new std::vector<std::shared_ptr<const float>>);
		tensorData->resize(model->numTensors());
	}
	// find the model input
	auto modelInputs = model->getInputs();
	if (modelInputs.size() != 1) {
		outWarningMessage = STR("We only support models with a single input, the current model has " << modelInputs.size() << " inputs");
		return false;
	}
	// resize the source image
	if (!(*tensorData.get())[modelInputs[0]]) {
		assert(inputShape.size()==3);
		TensorShape requiredShape = model->getTensorShape(modelInputs[0]);

		// adjust the required shape to the form [H,W,C]
		if (requiredShape.size() == 4) { // assume [B,H,W,C]
			if (requiredShape[0] != 1) {
				outWarningMessage = STR("Model's required shape " << requiredShape << " has 4 elements but doesn't begin with B=1,"
				                        " don't know how to adjust the image for it");
				return false;
			}
			requiredShape = tensorGetLastDims(requiredShape, 3);
		} else if (requiredShape.size() == 3) {
			if (requiredShape[0] == 1) { // assume [B=1,H,W], remove B and add C=1 for monochrome image
				requiredShape = tensorGetLastDims(requiredShape, 2);
				requiredShape.push_back(1);
			} else { // see if the shape is image-like
				if (requiredShape[2]!=1 && requiredShape[2]!=3) { // expect C=1 or C=3, otherwise we can't handle it
					outWarningMessage = STR("Model's required shape " << requiredShape << " has 3 elements but has C=1 or C=3,"
					                        " it doesn't look like it describes an image,"
					                        " don't know how to adjust the image for it");
					return false;
				}
			}
		} else {
			outWarningMessage = STR("Model's required shape " << requiredShape << " isn't standard, don't know how to adjust the image for it");
			return false;
		}

		// now we have requiredShape=[H,W,C], resize the image
		auto &sharedPtrInput = (*tensorData.get())[modelInputs[0]];
		if (inputShape != requiredShape)
			sharedPtrInput.reset(Image::resizeImage(inputTensor.get(), inputShape, requiredShape));
		else
			sharedPtrInput = inputTensor;
	}

	return true; // success
}

}
