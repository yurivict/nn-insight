// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#include <vector>
#include <array>

namespace Palette {
	typedef std::vector<std::array<float,3>> Colors;

	// all functions return object ownership
	Colors* ade20k_150colors();
};
