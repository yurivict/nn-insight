// Copyright (C) 2022 by Yuri Victorovich. All rights reserved.

#pragma once

#include "plugin-interface.h"
#include "tensor.h"

#include <string>
#include <vector>
#include <array>
#include <map>
#include <memory>
#include <functional>


namespace Compute {

bool buildComputeInputs(
	const PluginInterface::Model *model,
	std::array<unsigned,4> imageRegion,
	std::tuple<InputNormalizationRange,InputNormalizationColorOrder> inputNormalization,
	std::shared_ptr<float> &inputTensor, const TensorShape &inputShape,
	std::map<PluginInterface::TensorId, std::shared_ptr<const float>> &inputs, // output the set of inputs
	std::function<void(PluginInterface::TensorId)> cbTensorComputed,
	std::function<void(const std::string&)> cbWarningMessage
);

void fillInputs(
	std::map<PluginInterface::TensorId, std::shared_ptr<const float>> &inputs,
	std::unique_ptr<std::vector<std::shared_ptr<const float>>> &tensorData
);

bool compute(
	const PluginInterface::Model *model,
	std::unique_ptr<std::vector<std::shared_ptr<const float>>> &tensorData,
	std::function<void(PluginInterface::TensorId)> cbTensorComputed,
	std::function<void(const std::string&)> cbWarningMessage
);

}
