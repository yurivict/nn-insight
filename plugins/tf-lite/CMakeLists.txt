# Copyright (C) 2022 by Yuri Victorovich. All rights reserved.

set(SCHEMA_DEFINITION ${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/tensorflow/lite/schema/schema_v3b.fbs)

add_custom_target(schema_header
	COMMAND flatc --cpp ${SCHEMA_DEFINITION}
)

add_library(tf-lite-plugin SHARED
	tf-lite.cpp
)
set_target_properties(tf-lite-plugin PROPERTIES PREFIX "")
add_dependencies(tf-lite-plugin schema_header)
install(TARGETS tf-lite-plugin DESTINATION libexec/nn-insight)
