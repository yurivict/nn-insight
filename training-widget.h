// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#pragma once

#include <QComboBox>
#include <QGridLayout>
#include <QGroupBox>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>
#include <QWidget>

#include <memory>

class TrainingWidget : public QWidget {
	Q_OBJECT

	QGridLayout                 layout;
	QLabel                      trainingTypeLabel;
	QComboBox                   trainingTypeComboBox;
	QGroupBox                   dataSetGroupBox;
	QVBoxLayout                   dataSetLayout;
	std::unique_ptr<QWidget>      dataSetWidget; // dpeends on the training type
	QPushButton                 trainButton;

public:
	TrainingWidget(QWidget *parent);
	~TrainingWidget();

private:
	void updateTrainingType();
};
