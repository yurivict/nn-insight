#pragma once

#include <QPushButton>
#include <QByteArray>
class QResizeEvent;

class SvgPushButton : public QPushButton {

public: // constructor
	SvgPushButton(QByteArray svg, QWidget *parent);

public: // overridden
	void resizeEvent(QResizeEvent *event) override;
};
