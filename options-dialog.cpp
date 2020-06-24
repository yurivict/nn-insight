// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.


#include "options-dialog.h"

#include <QDoubleValidator>

#include <limits>

OptionsDialog::OptionsDialog(Options &options_, QWidget *parent)
: QDialog(parent)
, options(options_)
, layout(this)
, nearZeroCoefficientLabel(tr("Near Zero Coefficient"), this)
, nearZeroCoefficientEditBox(this)
, buttonBox(QDialogButtonBox::Ok, Qt::Horizontal, this)
{
	// add widgets to layouts
	layout.addWidget(&nearZeroCoefficientLabel,                  0/*row*/, 0/*col*/, 1/*rowSpan*/, 1/*columnSpan*/);
	layout.addWidget(&nearZeroCoefficientEditBox,                0/*row*/, 1/*col*/, 1/*rowSpan*/, 1/*columnSpan*/);
	layout.addWidget(&buttonBox,                                 1/*row*/, 1/*col*/, 1/*rowSpan*/, 2/*columnSpan*/);

	// set values
	nearZeroCoefficientEditBox.setText(QString("%1").arg(options.getNearZeroCoefficient()));

	// tooltips
	for (auto w : {(QWidget*)&nearZeroCoefficientLabel,(QWidget*)&nearZeroCoefficientEditBox})
		w->setToolTip(tr("Coefficient determining what values are considered to be near-zero. It is multiplied by a maximum of the absolute values of the value range."));

	// validators
	nearZeroCoefficientEditBox.setValidator(new QDoubleValidator(std::numeric_limits<double>::min(), std::numeric_limits<double>::max(), 3/*decimals*/, this));

	// connect signals
	connect(&nearZeroCoefficientEditBox, &QLineEdit::textChanged, [this](const QString &text) {
		options.setNearZeroCoefficient(text.toDouble());
	});
	connect(&buttonBox, SIGNAL(accepted()), this, SLOT(accept()));
}

