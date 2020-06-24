// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#pragma once


//
// OptionsDialog shows and allows the user to edit the Options structure.
//

#include "options.h"

#include <QDialog>
#include <QDialogButtonBox>
#include <QLineEdit>
#include <QGridLayout>
#include <QLabel>

class OptionsDialog : public QDialog {
	Q_OBJECT

	Options                           &options; // structure that we edit

	QGridLayout                       layout;
	QLabel                            nearZeroCoefficientLabel;
	QLineEdit                         nearZeroCoefficientEditBox;
	QDialogButtonBox                  buttonBox;

public:
	OptionsDialog(Options &options_, QWidget *parent);
};
