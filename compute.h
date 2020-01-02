// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#pragma once

#include "plugin-interface.h"
#include "nn-types.h"

#include <string>
#include <vector>
#include <array>
#include <memory>
#include <functional>


namespace Compute {

bool compute(
	const PluginInterface::Model *model,
	std::array<unsigned,4> imageRegion,
	std::tuple<InputNormalizationRange,InputNormalizationColorOrder> inputNormalization,
	std::shared_ptr<float> &inputTensor, const TensorShape &inputShape,
	std::unique_ptr<std::vector<std::shared_ptr<const float>>> &tensorData,
	std::function<void(const std::string&)> cbWarningMessage,
	std::function<void(PluginInterface::TensorId)> cbTensorDone);

}
