// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#include "misc.h" // has appSettings
#include "options.h"

#include <QSettings>

/// constr

Options::Options()
: nearZeroCoefficient(appSettings.value("Options.nearZeroCoefficient", 0.000001).toFloat())
{
}

/// static options object

Options& Options::get() {
	static Options options; // it is initialized after global static objects
	return options;
}

/// set-interface

void Options::setNearZeroCoefficient(float val) {
	nearZeroCoefficient = val;
	appSettings.setValue(QString("Options.nearZeroCoefficient"), val);
}

