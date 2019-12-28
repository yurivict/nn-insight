#pragma once

#include <QWidget>
#include <QHBoxLayout>
#include <QLabel>
#include <QSpinBox>

class ScaleImageWidget : public QWidget {
	Q_OBJECT

	QHBoxLayout       layout;
	QLabel            widthLabel;
	QSpinBox          widthSpinBox;
	QLabel            heightLabel;
	QSpinBox          heightSpinBox;
	QLabel            unitsLabel;

	bool              self; // to prevent signals from programmatically changed values

public: // constructor
	ScaleImageWidget(QWidget *parent);

public: // const values
	const static unsigned maxValue = 1000;

public: // interface
	void setFactor(unsigned factor); // factor is percentage as integer, same for witdth/height when set

signals:
	void scalingFactorChanged(unsigned widthFactor, unsigned heightFactor);

private: // internals
	void emitSignal();
};

