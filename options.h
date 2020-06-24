// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#pragma once

class Options {

	float       nearZeroCoefficient; // a coefficient that defines what "near-zero" is

public: // constr
	Options();

public: // static options object
	static Options& get();

public: // get-interface
	float       getNearZeroCoefficient() const {return nearZeroCoefficient;}

private: // set-interface
	void        setNearZeroCoefficient(float val);

	friend class OptionsDialog;
};

