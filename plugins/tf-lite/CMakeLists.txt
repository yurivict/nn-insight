add_library(tf-lite-plugin SHARED
	tf-lite.cpp
)
install(TARGETS tf-lite-plugin DESTINATION libexec/nn-insight)
