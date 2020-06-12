// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#include "misc.h"
#include "training-widget.h"
#include "util.h"

#include <QLineEdit>
#include <QGridLayout>
#include <QLabel>
#include <QSettings>
#include <QSpinBox>

#include <vector>
#include <string>

#include <exprtk.hpp>

/// local types

enum TrainingType {
	TrainingType_FunctionApproximationByFormula,
	TrainingType_FunctionApproximationFromTabulatedData,
	TrainingType_ImageLabeling
};

/// Training dataset type widgets

class DataSet_FunctionApproximationByFormulaWidget : public QWidget {
private: // types
	typedef float T;
	typedef exprtk::symbol_table<T> symbol_table_t;
	typedef exprtk::expression<T>     expression_t;
	typedef exprtk::parser<T>             parser_t;

private: // data
	QGridLayout        layout;
	QLabel             formulaLabel;
	QLineEdit          formulaEdit;
	QLabel             formulaErrorExcl;
	QLabel             argumentCountLabel;
	QSpinBox           argumentCountSpinBox;

	symbol_table_t           symbolTable;
	std::vector<std::string> symbolNames;
	std::vector<T>           symbolValues;
	expression_t             expression;

public:
	DataSet_FunctionApproximationByFormulaWidget(QWidget *parent)
	: QWidget(parent)
	, layout(this)
	, formulaLabel(tr("Formula"), this)
	, formulaEdit(this)
	, formulaErrorExcl("‼", this)
	, argumentCountLabel(tr("Num. Args"), this)
	, argumentCountSpinBox(this)
	{
		// add widgets to layouts
		layout.addWidget(&formulaLabel,         0,   0/*column*/);
		layout.addWidget(&formulaEdit,          0,   1/*column*/);
		layout.addWidget(&formulaErrorExcl,     0,   2/*column*/);
		layout.addWidget(&argumentCountLabel,   0,   3/*column*/);
		layout.addWidget(&argumentCountSpinBox, 0,   4/*column*/);

		// alignment
		formulaLabel.setAlignment(Qt::AlignRight|Qt::AlignVCenter);

		// widget states
		formulaEdit.setText(appSettings.value("TrainingWidget.DataSet_FunctionApproximationByFormulaWidget.formula", QVariant("sin(a*b)+tan(a-b)*cos(a-b)")).toString());
		formulaErrorExcl.setStyleSheet("QLabel {color: red;}");
		argumentCountSpinBox.setRange(1,32); // limited by the alphabet
		argumentCountSpinBox.setValue(appSettings.value("TrainingWidget.DataSet_FunctionApproximationByFormulaWidget.numArgs", QVariant(2)).toUInt());

		// signals
		connect(&formulaEdit, &QLineEdit::textChanged, [this](const QString &text) {
			checkFormula(text);
		});
		connect(&argumentCountSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), [this](int i) {
			updateNumArguments();
			checkFormula();
		});

		// update
		updateNumArguments();
		checkFormula();
	}
	~DataSet_FunctionApproximationByFormulaWidget() {
		appSettings.setValue("TrainingWidget.DataSet_FunctionApproximationByFormulaWidget.formula", formulaEdit.text());
		appSettings.setValue("TrainingWidget.DataSet_FunctionApproximationByFormulaWidget.numArgs", argumentCountSpinBox.value());
	}

private:
	void checkFormula() {
		checkFormula(formulaEdit.text());
	}
	void checkFormula(const QString &text) {
		parser_t parser;
		if (!parser.compile(Q2S(text), expression))
			formulaErrorExcl.show();
		else
			formulaErrorExcl.hide();
	}
	void updateNumArguments() {
		unsigned numArgs = argumentCountSpinBox.value();

		symbolTable = symbol_table_t();
		expression = expression_t();
		symbolNames .resize(numArgs);
		symbolValues.resize(numArgs);

		char argName = 'a';
		for (unsigned i = 0; i < numArgs; i++, argName++) {
			auto argNameStr = STR(argName);
			symbolNames[i] = argNameStr;
			symbolTable.add_variable(argNameStr.c_str(), symbolValues[i]);
		}
		expression.register_symbol_table(symbolTable);
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
		l->setToolTip(tr("Choose the training type to perform"));

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