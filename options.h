// Copyright (C) 2022 by Yuri Victorovich. All rights reserved.

#pragma once

class Options {

	bool        closeModelForTrainingModel;
	float       nearZeroCoefficient; // a coefficient that defines what "near-zero" is

public: // constr
	Options();

public: // static options object
	static Options& get();

public: // get-interface
	bool        getCloseModelForTrainingModel() const {return closeModelForTrainingModel;}
	float       getNearZeroCoefficient() const {return nearZeroCoefficient;}

private: // set-interface
	void        setCloseModelForTrainingModel(bool val);
	void        setNearZeroCoefficient(float val);

	friend class OptionsDialog;
};

