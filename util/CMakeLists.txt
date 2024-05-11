#author : Serhat Istin

include_directories( ${ROOT_INCLUDE_DIRS} ../src)

add_executable( mini_runner.exe runner.cpp )
target_link_libraries( mini_runner.exe MiniAnalysis )

add_executable( paint_mini.exe paint.cpp )
target_link_libraries(paint_mini.exe ${ROOT_LIBRARIES})



install(TARGETS paint_mini.exe mini_runner.exe  DESTINATION ${PROJECT_INSTALL_DIR}/bin)
install(
    FILES runAll.sh
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
    DESTINATION ${PROJECT_INSTALL_DIR}/bin
)

write_file(setup.sh "export PATH=\$\{PATH\}:${PROJECT_INSTALL_DIR}/bin" APPEND)
