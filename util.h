#pragma once

#include <string>
#include <QString>

#include <string>
#include <ostream>
#include <sstream>

class QWidget;

namespace Util {

std::string QStringToStlString(const QString &qs);
bool warningOk(QWidget *parent, const char *msg);
int getScreenDPI();

template <class Container, class Conv>
void splitString(const std::string& str, Container &cont, char delim = ' ') {
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
