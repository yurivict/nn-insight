
#include "util.h"
#include "misc.h"

#include <QApplication>
#include <QScreen>
#include <QCursor>
#include <QString>
#include <QMessageBox>

#include <limits>

namespace Util {

std::string QStringToStlString(const QString &qs) {
	return std::string(qs.toUtf8().constData());
}

bool warningOk(QWidget *parent, const char *msg) {
  QMessageBox::warning(parent, "Warning", msg, QMessageBox::Ok);
  return false; // for convenience of callers
}

int getScreenDPI() {
	return QApplication::screens().at(0)->logicalDotsPerInch();
}

QPoint getGlobalMousePos() {
	return QCursor::pos(QApplication::screens().at(0));
}

std::string formatUIntHumanReadable(size_t u) {
	if (u <= 999)
		return STR(u);
	else {
		auto ddd = STR(u%1000);
		while (ddd.size() < 3)
			ddd = std::string("0")+ddd;
		return STR(formatUIntHumanReadable(u/1000) << "," << ddd);
	}
}

std::tuple<float,float> arrayMinMax(const float *arr, size_t len) {
	float amin = std::numeric_limits<float>::max();
	float amax = std::numeric_limits<float>::lowest();
	for (const float *ae = arr + len; arr < ae; arr++) {
		if (*arr < amin)
			amin = *arr;
		if (*arr > amax)
			amax = *arr;
	}
	return std::tuple<float,float>(amin, amax);
}

}

