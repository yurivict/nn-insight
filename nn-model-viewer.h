#pragma once

#include <QWidget>
#include <QSplitter>
#include <memory>
#include "plugin-interface.h"

class NnModelViewer : public QWidget {
  Q_OBJECT

  std::unique_ptr<PluginInterface> pluginInterface;
  unsigned                         numModel;

  QSplitter       splitter;  // we place other widgets into a splitter, begin with a single model structure-viewer

public:
  NnModelViewer(QWidget *parent, PluginInterface *pluginInterface_, unsigned numModel_);
};
