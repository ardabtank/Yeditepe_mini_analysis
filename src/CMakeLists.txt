add_library(MiniAnalysis SHARED tree_reader.cpp)
include_directories( ${ROOT_INCLUDE_DIRS} )
target_link_libraries( MiniAnalysis ${ROOT_LIBRARIES} )

install(TARGETS MiniAnalysis  DESTINATION ${PROJECT_INSTALL_DIR}/lib )