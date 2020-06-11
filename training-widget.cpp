// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#include "misc.h"
#include "training-widget.h"

#include <QLineEdit>
#include <QGridLayout>
#include <QLabel>
#include <QSettings>

/// local types

enum TrainingType {
	TrainingType_FunctionApproximationByFormula,
	TrainingType_FunctionApproximationFromTabulatedData,
	TrainingType_ImageLabeling
};

/// Training dataset type widgets

class DataSet_FunctionApproximationByFormulaWidget : public QWidget {
	QGridLayout        layout;
	QLabel             formulaLabel;
	QLineEdit          formulaEdit;
public:
	DataSet_FunctionApproximationByFormulaWidget(QWidget *parent)
	: QWidget(parent)
	, layout(this)
	, formulaLabel(tr("Formula"), this)
	, formulaEdit(this)
	{
		// add widgets to layouts
		layout.addWidget(&formulaLabel,     0,   0/*column*/);
		layout.addWidget(&formulaEdit,      0,   1/*column*/);

		// alignment
		formulaLabel.setAlignment(Qt::AlignRight|Qt::AlignVCenter);

		// widget states
		formulaEdit.setText(appSettings.value("TrainingWidget.DataSet_FunctionApproximationByFormulaWidget.formula", QVariant("sin(a*b)+tan(a-b)*cos(a-b)")).toString());
	}
	~DataSet_FunctionApproximationByFormulaWidget() {
		appSettings.setValue("TrainingWidget.DataSet_FunctionApproximationByFormulaWidget.formula", formulaEdit.text());
	}
};

class DataSet_FunctionApproximationFromTabulatedData : public QWidget {
	QGridLayout        layout;
	QLabel             xxxLabel;
public:
	DataSet_FunctionApproximationFromTabulatedData(QWidget *parent)
	: QWidget(parent)
	, layout(this)
	, xxxLabel(tr("FunctionApproximationFromTabulatedData"), this)
	{
		// add widgets to layouts
		layout.addWidget(&xxxLabel,     0,   0/*column*/);
	}
}
;
class DataSet_ImageLabeling : public QWidget {
	QGridLayout        layout;
	QLabel             xxxLabel;
public:
	DataSet_ImageLabeling(QWidget *parent)
	: QWidget(parent)
	, layout(this)
	, xxxLabel(tr("ImageLabeling"), this)
	{
		// add widgets to layouts
		layout.addWidget(&xxxLabel,     0,   0/*column*/);
	}
};

/// TrainingWidget methods

TrainingWidget::TrainingWidget(QWidget *parent)
: QWidget(parent)
, layout(this)
, trainingTypeLabel("Type of Training", this)
, trainingTypeComboBox(this)
, dataSetGroupBox(tr("Training Data"), this)
,   dataSetLayout(&dataSetGroupBox)
, trainButton(tr("Start Training"), this)
{
	// add widgets to layouts
	layout.addWidget(&trainingTypeLabel,     0,   0/*column*/);
	layout.addWidget(&trainingTypeComboBox,  0,   1/*column*/);
	layout.addWidget(&dataSetGroupBox,       1,   0/*column*/,  1/*rowSpan*/, 2/*columnSpan*/);
	layout.addWidget(&trainButton,           2,   0/*column*/,  1/*rowSpan*/, 2/*columnSpan*/);

	// fill comboboxes
	trainingTypeComboBox.addItem(tr("Function approximation (by formula)"), TrainingType_FunctionApproximationByFormula);
	trainingTypeComboBox.addItem(tr("Function approximation (from tabulated data)"), TrainingType_FunctionApproximationFromTabulatedData);
	trainingTypeComboBox.addItem(tr("Image labeling"), TrainingType_ImageLabeling);

	// alignment
	trainingTypeLabel.setAlignment(Qt::AlignRight|Qt::AlignVCenter);

	// widget states
	trainingTypeComboBox.setCurrentIndex(trainingTypeComboBox.findData(appSettings.value("TrainingWidget.trainingType", QVariant(TrainingType_FunctionApproximationByFormula))));

	// tooltips
	for (auto l : {(QWidget*)&trainingTypeLabel,(QWidget*)&trainingTypeComboBox})
		l->           setToolTip(tr("Choose the training type to perform"));

	// process signals
	connect(&trainingTypeComboBox, QOverload<int>::of(&QComboBox::activated), [this](int index) {
		appSettings.setValue("TrainingWidget.trainingType", trainingTypeComboBox.itemData(index));
		updateTrainingType();
	});

	// set data set widget initially
	updateTrainingType();
}

TrainingWidget::~TrainingWidget() {
}

// privates

void TrainingWidget::updateTrainingType() {
	switch ((TrainingType)trainingTypeComboBox.itemData(trainingTypeComboBox.currentIndex()).toUInt()) {
	case TrainingType_FunctionApproximationByFormula:
		dataSetWidget.reset(new DataSet_FunctionApproximationByFormulaWidget(&dataSetGroupBox));
		break;
	case TrainingType_FunctionApproximationFromTabulatedData:
		dataSetWidget.reset(new DataSet_FunctionApproximationFromTabulatedData(&dataSetGroupBox));
		break;
	case TrainingType_ImageLabeling:
		dataSetWidget.reset(new DataSet_ImageLabeling(&dataSetGroupBox));
		break;
	}
	dataSetLayout.addWidget(dataSetWidget.get());
}
