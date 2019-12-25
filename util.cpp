
#include "util.h"
#include "misc.h"

#include <QApplication>
#include <QScreen>
#include <QCursor>
#include <QString>
#include <QMessageBox>

#include <limits>
#include <cstring>

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

std::string formatUIntHumanReadableSuffixed(size_t u) {
	auto one = [](size_t u, size_t degree, char chr) {
		auto du = u/degree;
		if (du >= 10)
			return STR(formatUIntHumanReadable(du) << ' ' << chr);
		else
			return STR(formatUIntHumanReadable(du) << '.' << (u%degree)/(degree/10) << ' ' << chr);
	};
	if (u >= 1000000000000) // in Tera-range
		return one(u, 1000000000000, 'T');
	if (u >= 1000000000) // in Giga-range
		return one(u, 1000000000, 'G');
	else if (u >= 1000000) // in Mega-range
		return one(u, 1000000, 'M');
	else if (u >= 1000) // in kilo-range
		return one(u, 1000, 'k');
	return STR(u << ' '); // because it is followed by the unit name

}

std::string formatFlops(size_t flops) {
	return STR(formatUIntHumanReadableSuffixed(flops) << "flops");
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

float* copyFpArray(const float *a, size_t sz) {
	auto n = new float[sz];
	std::memcpy(n, a, sz*sizeof(float));
	return n;
}

}

