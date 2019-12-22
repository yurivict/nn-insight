
#include "nn-model-viewer.h"


NnModelViewer::NnModelViewer(QWidget *parent, PluginInterface *pluginInterface_, unsigned numModel_)
: QWidget(parent)
, pluginInterface(pluginInterface_) // assumed to be passed in the opened condition so that nothing can fail (XXX really?)
, numModel(numModel_)
{
}
