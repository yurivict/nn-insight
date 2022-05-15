// Copyright (C) 2022 by Yuri Victorovich. All rights reserved.

#pragma once

#include "plugin-interface.h"

#include <ostream>

namespace ModelValidator {

bool validate(const PluginInterface::Model *model, std::ostream &os);

}
