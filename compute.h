#pragma once

#include "plugin-interface.h"
#include "nn-types.h"

#include <string>
#include <vector>
#include <memory>

namespace Compute {

bool compute(
	const PluginInterface::Model *model,
	std::shared_ptr<float> &inputTensor, const TensorShape &inputShape,
	std::unique_ptr<std::vector<std::shared_ptr<const float>>> &tensorData,
	std::string &outWarningMessage);

}
