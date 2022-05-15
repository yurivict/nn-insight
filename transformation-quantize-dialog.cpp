// Copyright (C) 2022 by Yuri Victorovich. All rights reserved.

#include "misc.h"

#include "transformation-quantize-dialog.h"

#include <QIntValidator>
#include <QPushButton>
#include <QSettings>
#include <QVariant>


TransformationQuantizeDialog::TransformationQuantizeDialog(QWidget *parent)
: QDialog(parent)
, weightsQuantizationSegments(appSettings.value("TransformationQuantizeDialog.weightsQuantizationSegments", QVariant(1000)/*default*/).toUInt())
, biasesQuantizationSegments(appSettings.value("TransformationQuantizeDialog.biasesQuantizationSegments", QVariant(1000)/*default*/).toUInt())
, layout(this)
, quantizeWeightsCheckBox(tr("Quantize weights"), this)
, quantizeWeightsEditBox(this)
, quantizeWeightsLabel(tr("parts"), this)
, quantizeBiasesCheckBox(tr("Quantize biases"), this)
, quantizeBiasesEditBox(this)
, quantizeBiasesLabel(tr("parts"), this)
, buttonBox(QDialogButtonBox::Ok|QDialogButtonBox::Cancel, Qt::Horizontal, this)
{
	// add widgets to the layout
	layout.addWidget(&quantizeWeightsCheckBox, 0/*row*/, 0/*col*/, 1/*rowSpan*/, 1/*columnSpan*/);
	layout.addWidget(&quantizeWeightsEditBox,  0/*row*/, 1/*col*/, 1/*rowSpan*/, 1/*columnSpan*/);
	layout.addWidget(&quantizeWeightsLabel,    0/*row*/, 2/*col*/, 1/*rowSpan*/, 1/*columnSpan*/);
	layout.addWidget(&quantizeBiasesCheckBox,  1/*row*/, 0/*col*/, 1/*rowSpan*/, 1/*columnSpan*/);
	layout.addWidget(&quantizeBiasesEditBox,   1/*row*/, 1/*col*/, 1/*rowSpan*/, 1/*columnSpan*/);
	layout.addWidget(&quantizeBiasesLabel,     1/*row*/, 2/*col*/, 1/*rowSpan*/, 1/*columnSpan*/);
	layout.addWidget(&buttonBox,               2/*row*/, 0/*col*/, 2/*rowSpan*/, 3/*columnSpan*/);

	// set default widget state
	quantizeWeightsCheckBox.setChecked(true); // checkboxes are selected by default
	quantizeBiasesCheckBox .setChecked(true);
	quantizeWeightsEditBox .setText(QString("%1").arg(weightsQuantizationSegments));
	quantizeBiasesEditBox  .setText(QString("%1").arg(biasesQuantizationSegments));

	// set validators
	quantizeWeightsEditBox.setValidator(new QIntValidator(1/*minimum*/, std::numeric_limits<int>::max(), this)); // affected by https://bugreports.qt.io/browse/QTBUG-84558
	quantizeBiasesEditBox .setValidator(new QIntValidator(1/*minimum*/, std::numeric_limits<int>::max(), this));

	// tooltips
	for (QWidget *w : {(QWidget*)&quantizeWeightsCheckBox,(QWidget*)&quantizeWeightsEditBox,(QWidget*)&quantizeWeightsLabel})
		w->setToolTip(tr("Please specify a number of segments that the range of values in all weight arrays would be quantized into"));
	for (QWidget *w : {(QWidget*)&quantizeBiasesCheckBox,(QWidget*)&quantizeBiasesEditBox,(QWidget*)&quantizeBiasesLabel})
		w->setToolTip(tr("Please specify a number of segments that the range of values in all bias arrays would be quantized into"));

	// connect signals
	connect(&quantizeWeightsCheckBox, &QCheckBox::stateChanged, [this](int) {
		quantizeWeightsEditBox.setEnabled(quantizeWeightsCheckBox.isChecked());
		quantizeWeightsLabel  .setEnabled(quantizeWeightsCheckBox.isChecked());
		buttonBox.button(QDialogButtonBox::Ok)->setEnabled(quantizeWeightsCheckBox.isChecked() || quantizeBiasesCheckBox.isChecked());
	});
	connect(&quantizeWeightsEditBox, &QLineEdit::textChanged, [this](const QString &text) {
		weightsQuantizationSegments = text.toUInt();
	});
	connect(&quantizeBiasesCheckBox, &QCheckBox::stateChanged, [this](int) {
		quantizeBiasesEditBox.setEnabled(quantizeBiasesCheckBox.isChecked());
		quantizeBiasesLabel  .setEnabled(quantizeBiasesCheckBox.isChecked());
		buttonBox.button(QDialogButtonBox::Ok)->setEnabled(quantizeWeightsCheckBox.isChecked() || quantizeBiasesCheckBox.isChecked());
	});
	connect(&quantizeBiasesEditBox, &QLineEdit::textChanged, [this](const QString &text) {
		biasesQuantizationSegments = text.toUInt();
	});
	connect(&buttonBox, SIGNAL(accepted()), this, SLOT(accept()));
	connect(&buttonBox, SIGNAL(rejected()), this, SLOT(reject()));
}

TransformationQuantizeDialog::~TransformationQuantizeDialog() {
	// save into settings
	appSettings.setValue("TransformationQuantizeDialog.weightsQuantizationSegments", QVariant(weightsQuantizationSegments));
	appSettings.setValue("TransformationQuantizeDialog.biasesQuantizationSegments", QVariant(biasesQuantizationSegments));
}
