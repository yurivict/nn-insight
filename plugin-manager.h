// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#pragma once

#include <string>

// forward decls
class PluginInterface;

// define the library interface
typedef PluginInterface*  (*PluginLibInterface)();

namespace PluginManager {

class Plugin; // publicly opaque class

const Plugin* loadPlugin(const std::string &pluginName);
PluginLibInterface getInterface(const Plugin *plugin);
void unloadPlugin(const Plugin *plugin);

} // PluginManager
