
#include "util.h"

#include <QApplication>
#include <QScreen>
#include <QString>
#include <QMessageBox>

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

}

