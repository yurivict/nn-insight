// Copyright (C) 2022 by Yuri Victorovich. All rights reserved.

#pragma once

#include <QCheckBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QLineEdit>
#include <QGridLayout>
#include <QLabel>

class TransformationQuantizeDialog : public QDialog {
	Q_OBJECT

public:
	TransformationQuantizeDialog(QWidget *parent);
	~TransformationQuantizeDialog();

private: // values
	unsigned           weightsQuantizationSegments;
	unsigned           biasesQuantizationSegments;

private: // fields
	QGridLayout                       layout;
	QCheckBox                         quantizeWeightsCheckBox;
	QLineEdit                         quantizeWeightsEditBox;
	QLabel                            quantizeWeightsLabel;
	QCheckBox                         quantizeBiasesCheckBox;
	QLineEdit                         quantizeBiasesEditBox;
	QLabel                            quantizeBiasesLabel;
	QDialogButtonBox                  buttonBox;

public: // access
	bool doWeightsQuantization() const {return quantizeWeightsCheckBox.isChecked();}
	bool doBiasesQuantization() const {return quantizeBiasesCheckBox.isChecked();}
	unsigned getWeightsQuantizationSegments() const {return weightsQuantizationSegments;}
	unsigned getBiasesQuantizationSegments() const {return biasesQuantizationSegments;}
};
