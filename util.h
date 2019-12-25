#pragma once

#include <string>
#include <QString>
#include <QPoint>

#include <string>
#include <vector>
#include <ostream>
#include <sstream>
#include <tuple>

class QWidget;

namespace Util {

std::string QStringToStlString(const QString &qs);
bool warningOk(QWidget *parent, const char *msg);
int getScreenDPI();
QPoint getGlobalMousePos();
std::string formatUIntHumanReadable(size_t u);
std::string formatUIntHumanReadableSuffixed(size_t u);
std::string formatFlops(size_t flops);
std::tuple<float,float> arrayMinMax(const float *arr, size_t len);
float* copyFpArray(const float *a, size_t sz);

inline void splitString(const std::string& str, std::vector<std::string> &cont, char delim = ' ') {
	std::stringstream ss(str);
	std::string token;
	while (std::getline(ss, token, delim))
		cont.push_back(token);
}

template <class Container, class Conv>
inline void splitString(const std::string& str, Container &cont, char delim = ' ') {
	std::stringstream ss(str);
	std::string token;
	while (std::getline(ss, token, delim))
		cont.push_back(Conv::conv(token));
}

};

#define Q2S(qs) Util::QStringToStlString(qs)
#define S2Q(ss) QString(ss.c_str())

// general templates

template<typename T, unsigned long N>
inline std::ostream& operator<<(std::ostream &os, const std::array<T,N> &c) {
	os << "[";
	unsigned idx = 0;
	for (auto &e : c) {
		if (idx++ != 0)
			os << ',';
		os << e;
	}
	os << "]";
	return os;
}

template<typename T>
inline std::ostream& operator<<(std::ostream &os, const std::vector<T> &c) {
	os << "[";
	unsigned idx = 0;
	for (auto &e : c) {
		if (idx++ != 0)
			os << ',';
		os << e;
	}
	os << "]";
	return os;
}
