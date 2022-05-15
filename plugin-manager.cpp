// Copyright (C) 2022 by Yuri Victorovich. All rights reserved.

#include "plugin-manager.h"
#include "plugin-interface.h"
#include "misc.h"
#include "util.h"

#include <dlfcn.h>

#include <string>
#include <map>
#include <memory>

namespace PluginManager {

class Plugin {
public:
	mutable int        refCnt;
	std::string        name;
	std::string        libraryPath;
	void*              handle;
	PluginLibInterface pluginLibInterface;

	Plugin(const std::string &name_, const std::string &libraryPath_, void *handle_, PluginLibInterface pluginLibInterface_)
	: refCnt(1)
	, name(name_)
	, libraryPath(libraryPath_)
	, handle(handle_)
	, pluginLibInterface(pluginLibInterface_)
	{ }

	const Plugin* ref() const {
		refCnt++;
		return this;
	}
	int unref() const {
		return --refCnt;
	}
};

// plugin registry
static std::map<std::string, std::unique_ptr<Plugin>> registry; // all loaded plugins


// local helpers

static std::string pluginNameToPluginLibraryPath(const std::string &pluginName) {
	auto getMyExecutionPath = []() {
		auto myExe = Util::getMyOwnExecutablePath();
		auto sep = myExe.find_last_of('/');
		if (sep==std::string::npos)
			FAIL("bad executable path: it doesn't contain a separator '/': " << myExe)
		return myExe.substr(0,sep);
	};

	// try locally built dir
	auto pathLocalDevDir = STR(getMyExecutionPath() << "/plugins/" << pluginName << "/" << pluginName << "-plugin.so");
	if (Util::doesFileExist(pathLocalDevDir.c_str()))
		return pathLocalDevDir;


	// try globally installed situation
	auto pathGlobalInstallDir = STR(getMyExecutionPath() << "/../libexec/nn-insight/" << pluginName << "-plugin.so");
	if (Util::doesFileExist(pathGlobalInstallDir.c_str()))
		return pathGlobalInstallDir;

	// fail to find it
	return "";
}

//
// interface implementation
//

const Plugin* loadPlugin(const std::string &pluginName) {
	// is it already loaded?
	{
		auto it = registry.find(pluginName);
		if (it != registry.end())
			return it->second.get()->ref();
	}

	// plugin name -> shared library object
	auto pluginLibraryPath = pluginNameToPluginLibraryPath(pluginName);
	if (pluginLibraryPath.empty())
		return nullptr; // failed to find the plugin

	auto handle = ::dlopen(pluginLibraryPath.c_str(), RTLD_NOW);
	if (handle == nullptr) {
		PRINT_ERR("Failed to instantiate the plugin '" << pluginName << "' to open the file " << pluginLibraryPath << "': " << ::dlerror())
		return nullptr;
	}

	// shared library object -> symbol
	auto pluginLibInterface = (PluginLibInterface)::dlsym(handle, "createPluginInterface");
	if (pluginLibInterface == nullptr) {
		auto err = STR("Failed to instantiate the plugin '" << pluginName << "' to open the file ;" << pluginLibraryPath << "': the symbol is missing");
		if (::dlclose(handle) != 0)
			err += STR("; failed to unload the plugin's shared library '" << pluginLibraryPath << "': " << ::dlerror());
		PRINT_ERR(err)
		return nullptr;
	}

	Plugin *plugin = new Plugin(pluginName, pluginLibraryPath, handle, pluginLibInterface);
	registry[pluginName].reset(plugin);

	return plugin;
}

PluginLibInterface getInterface(const Plugin *plugin) {
	return plugin->pluginLibInterface;
}

void unloadPlugin(const Plugin *plugin) {
	if (plugin->unref() == 0) {
		if (::dlclose(plugin->handle) != 0)
			PRINT_ERR("failed to unload the plugin's shared library '" << plugin->libraryPath << "': " << ::dlerror())
		registry.erase(plugin->name);
	}
}

}
