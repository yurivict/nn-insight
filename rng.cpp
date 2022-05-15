// Copyright (C) 2022 by Yuri Victorovich. All rights reserved.

#include "rng.h"


namespace Rng {


static std::random_device rd;
std::mt19937 generator(rd());

}
