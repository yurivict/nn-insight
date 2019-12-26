add_library(tf-lite-plugin SHARED
	tf-lite.cpp
)
set_target_properties(tf-lite-plugin PROPERTIES PREFIX "")
install(TARGETS tf-lite-plugin DESTINATION libexec/nn-insight)
