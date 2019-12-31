#pragma once

#include <QWidget>
#include <QGridLayout>
#include <QPushButton>

class NoNnIsOpenWidget : public QWidget {
	Q_OBJECT

	QGridLayout        layout;
	QPushButton        openNnFileButton;
	QWidget            spacer;

public:
	NoNnIsOpenWidget(QWidget *parent);

signals:
	void openNeuralNetworkFilePressed();
};
