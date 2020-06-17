// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#pragma once

#include "plugin-interface.h"

#include <QComboBox>
#include <QGridLayout>
#include <QGroupBox>
#include <QLabel>
#include <QPushButton>
#include <QThread>
#include <QVBoxLayout>
#include <QWidget>

#include <array>
#include <memory>
#include <vector>

class TrainingWidget : public QWidget {
	Q_OBJECT

// types
	enum TrainingType {
		TrainingType_FunctionApproximationByFormula,
		TrainingType_FunctionApproximationFromTabulatedData,
		TrainingType_ImageLabeling
	};

// data
	QGridLayout                 layout;
	QLabel                      trainingTypeLabel;
	QComboBox                   trainingTypeComboBox;
	QGroupBox                   dataSetGroupBox;
	QVBoxLayout                   dataSetLayout;
	std::unique_ptr<QWidget>      dataSetWidget; // depends on the training type
	QPushButton                 verifyDerivativesButton;
	QPushButton                 trainButton;

	TrainingType                trainingType;
	std::unique_ptr<QThread>    trainingThread;

	bool                        threadStopFlag;

public:
	TrainingWidget(QWidget *parent, PluginInterface::Model *model, float modelPendingTrainingDerivativesCoefficient);
	~TrainingWidget();

private:
	void updateTrainingType();
	std::array<std::vector<float>,2> getData(bool validation) const;
};
