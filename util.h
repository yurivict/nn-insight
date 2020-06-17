// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#pragma once

#include <string>
#include <QString>
#include <QPoint>
#include <QPixmap>
#include <QStringList>
#include <QImage>
#include <QByteArray>
#include <QSize>
#include <QPainter>
class QComboBox;

#include <string>
#include <vector>
#include <ostream>
#include <memory>
#include <sstream>
#include <tuple>
#include <algorithm>

class QWidget;

namespace Util {

std::string QStringToStlString(const QString &qs);
bool messageOk(QWidget *parent, const QString &title, const QString &msg);
bool warningOk(QWidget *parent, const QString &msg);
float getScreenDPI();
QPoint getGlobalMousePos();
std::string formatUIntHumanReadable(size_t u);
std::string formatUIntHumanReadableSuffixed(size_t u);
std::string formatFlops(size_t flops);
template<typename T>
std::tuple<T,T> arrayMinMax(const T *arr, size_t len) {
	T amin = std::numeric_limits<T>::max();
	T amax = std::numeric_limits<T>::lowest();
	for (const T *ae = arr + len; arr < ae; arr++) {
		if (*arr < amin)
			amin = *arr;
		if (*arr > amax)
			amax = *arr;
	}
	return std::tuple<T,T>(amin, amax);
}
template<typename T>
T* arrayOfOnes(unsigned sz) {
	std::unique_ptr<T> arr(new T[sz]);
	for (T *a = arr.get(), *ae = a+sz; a<ae; a++)
		*a = 1;
	return arr.release();
}

float* copyFpArray(const float *a, size_t sz);
size_t getFileSize(const QString &fileName);
QPixmap getScreenshot(bool hideOurWindows);
unsigned char* convertArrayFloatToUInt8(const float *a, size_t size);
bool doesFileExist(const char *filePath);
QStringList readListFromFile(const char *fileName);
std::string getMyOwnExecutablePath();
QImage svgToImage(const QByteArray& svgContent, const QSize& size, QPainter::CompositionMode mode);
void selectComboBoxItemWithItemData(QComboBox &comboBox, int value);
void setWidgetColor(QWidget *widget, const char *color);
std::string charToSubscript(char ch);
std::string stringToSubscript(const std::string &str);

template<typename T>
bool isValueIn(const std::vector<T> &v, T val) {
	return std::find(v.begin(), v.end(), val) != v.end();
}

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

// silence unwanted warnings about unused variables
#define UNUSED(expr) do { (void)(expr); } while (0);
