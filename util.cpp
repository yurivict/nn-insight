// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#include "util.h"
#include "misc.h"

#include <QApplication>
#include <QScreen>
#include <QCursor>
#include <QString>
#include <QMessageBox>
#include <QFile>
#include <QPixmap>
#include <QGuiApplication>
#include <QWindow>
#include <QStringList>
#include <QImage>
#include <QByteArray>
#include <QSize>
#include <QPainter>
#include <QSvgRenderer>
#include <QComboBox>

#include <limits>
#include <cstring>
#include <memory>

#include <unistd.h> // sleep,readlink
#include <sys/stat.h>
#include <assert.h>

namespace Util {

std::string QStringToStlString(const QString &qs) {
	return std::string(qs.toUtf8().constData());
}

bool messageOk(QWidget *parent, const QString &title, const QString &msg) {
	QMessageBox::warning(parent, title, msg, QMessageBox::Ok);
	return false; // for convenience of callers
}

bool warningOk(QWidget *parent, const QString &msg) {
	QMessageBox::warning(parent, "Warning", msg, QMessageBox::Ok);
	return false; // for convenience of callers
}

void centerWidgetAtOtherWidget(QWidget *widget, QWidget *otherWidget, float fraction) {
	widget->resize(otherWidget->size()*fraction);
	// TODO move should be also performed, but our uses get by because we use centerWidgetAtOtherWidget on modal dialogs before show() and the dialog gets centered automatically
}

float getScreenDPI() {
	static float dpi = QApplication::screens().at(0)->logicalDotsPerInch();
	return dpi;
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

float* copyFpArray(const float *a, size_t sz) {
	auto n = new float[sz];
	std::memcpy(n, a, sz*sizeof(float));
	return n;
}

size_t getFileSize(const QString &fileName) {
	size_t size = 0;
	QFile file(fileName);
	if (file.open(QIODevice::ReadOnly)) {
		size = file.size();
		file.close();
	} 
	return size;
}

QPixmap getScreenshot(bool hideOurWindows) {
	QScreen *screen = QGuiApplication::primaryScreen();

	std::vector<QWidget*> windowsToHide = hideOurWindows ? std::vector<QWidget*>{QApplication::activeWindow()} : std::vector<QWidget*>{};
	for (auto w : windowsToHide)
		w->hide();

	QApplication::beep(); // ha ha

	if (!windowsToHide.empty()) {
		QCoreApplication::processEvents();
		::sleep(1); // without this sleep screen doesn't have enough time to hide our window when complex windows are behind (like a browser)
	}

	auto pixmap = screen->grabWindow(0);

	for (auto w : windowsToHide)
		w->show();

	return pixmap;
}

unsigned char* convertArrayFloatToUInt8(const float *a, size_t size) { // ASSUME that a is normalized to 0..255
	std::unique_ptr<unsigned char> cc(new unsigned char[size]);

	auto c = cc.get();
	for (const float *ae = a+size; a<ae; )
		*c++ = *a++;

	return cc.release();
}

bool doesFileExist(const char *filePath) {
	struct stat s;
	return ::stat(filePath, &s)==0 && (s.st_mode&S_IFREG);
}

QStringList readListFromFile(const char *fileName) {
	QString data;
	QFile file(fileName);
	if (!file.open(QIODevice::ReadOnly))
		FAIL("failed to open the file " << fileName)
	data = file.readAll();
	file.close();
	return data.split("\n", QString::SkipEmptyParts);
}

std::string getMyOwnExecutablePath() {
#if defined(__FreeBSD__) || defined(__DragonFly__) || defined(__OpenBSD__) || defined(__NetBSD__)
	const char* selfExeLink = "/proc/curproc/file";
#elif defined(__linux__)
	const char* selfExeLink = "/proc/self/exe";
#else
#  error "Your OS is not yet supported"
#endif

	char buf[PATH_MAX+1];
	auto res = ::readlink(selfExeLink, buf, sizeof(buf) - 1);
	if (res == -1)
		FAIL("Failed to read the link " << selfExeLink << " to determine our executable path")
	buf[res] = 0;

	return buf;
}

QImage svgToImage(const QByteArray& svgContent, const QSize& size, QPainter::CompositionMode mode) {
	QImage image(size.width(), size.height(), QImage::Format_ARGB32);

	QPainter painter(&image);
	painter.setCompositionMode(mode);
	image.fill(Qt::transparent);
	QSvgRenderer(svgContent).render(&painter);

	return image;
}

void selectComboBoxItemWithItemData(QComboBox &comboBox, int value) {
	for (unsigned i=0, ie=comboBox.count(); i<ie; i++)
		if (comboBox.itemData(i).toInt() == value) {
			comboBox.setCurrentIndex(i);
			return;
		}
	assert(false); // item with itemData=value not found
}

void setWidgetColor(QWidget *widget, const char *color) {
	widget->setStyleSheet(S2Q(STR("color: " << color)));
}

std::string charToSubscript(char ch) {
	switch (ch) {
	case '0': return STR("₀");
	case '1': return STR("₁");
	case '2': return STR("₂");
	case '3': return STR("₃");
	case '4': return STR("₄");
	case '5': return STR("₅");
	case '6': return STR("₆");
	case '7': return STR("₇");
	case '8': return STR("₈");
	case '9': return STR("₉");
	case '+': return STR("₊");
	case '-': return STR("₋");
	case '=': return STR("=");
	case '(': return STR("₍");
	case ')': return STR("₎");
	case 'x': return STR("ₓ");
	default:
		assert(false);
		return " ";
	}
}

std::string stringToSubscript(const std::string &str) {
	std::ostringstream ss;
	for (auto ch : str)
		ss << charToSubscript(ch);
	return ss.str();
}

}

