// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#pragma once

#include "plugin-interface.h"
#include "training-progress-widget.h"

#include <QComboBox>
#include <QDoubleSpinBox>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSpinBox>
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
	QGroupBox                   parametersGroupBox;
	QHBoxLayout                   parametersLayout;
	QLabel                        paramBatchSizeLabel;
	QSpinBox                      paramBatchSizeSpinBox;
	QLabel                        paramLearningRateLabel;
	QDoubleSpinBox                paramLearningRateSpinBox;
	QLabel                        paramMaxBatchesLabel;
	QSpinBox                      paramMaxBatchesSpinBox;
	QLabel                        paramOptimizationAlgorithmLabel;
	QComboBox                     paramOptimizationAlgorithmComboBox;
	QPushButton                 verifyDerivativesButton;
	QPushButton                 trainButton;
	QLabel                      trainingStats;
	TrainingProgressWidget      trainingProgressWidget;

	TrainingType                trainingType;
	std::unique_ptr<QThread>    trainingThread;

	bool                        threadStopFlag;
	float                       minLoss;
	float                       maxLoss;

public:
	TrainingWidget(QWidget *parent, QWidget *topLevelWidget, PluginInterface::Model *model, float modelPendingTrainingDerivativesCoefficient);
	~TrainingWidget();

private:
	void updateTrainingType();
	std::array<std::vector<float>,2> getData(bool validation) const;
};
