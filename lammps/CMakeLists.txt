# min_style rgmin as a LAMMPS plugin ("plugin load rgminplugin.so").
#
#   cmake -S lammps -B build-lammps -DLAMMPS_SOURCE_DIR=<lammps>/src
#   cmake --build build-lammps
#
# The plugin compiles against the headers of the LAMMPS tree that built
# the lmp binary, with the same MPI and the same LAMMPS_SIZES setting.
# By default it builds rgmin's staticlib with cargo (--features capi)
# and links it in, so the plugin needs no rgmin library at run time.
cmake_minimum_required(VERSION 3.16)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "Configure min_style rgmin in a separate build directory")
endif()

project(rgmin_lammps VERSION 0.2.0 LANGUAGES C CXX)

set(LAMMPS_SOURCE_DIR "" CACHE PATH "The src directory of the LAMMPS tree that built lmp")
set(LAMMPS_SIZES "smallbig" CACHE STRING "LAMMPS integer sizes: smallbig, bigbig or smallsmall")
option(LAMMPS_MPI "LAMMPS was built with MPI (OFF: the STUBS serial library)" ON)
set(RGMIN_LIBRARY "" CACHE FILEPATH
  "Prebuilt librgmin.a or librgmin.so with the capi feature; empty builds it with cargo")
set(RGMIN_LAMMPS_LMP "" CACHE FILEPATH "lmp binary that ctest runs the plugin tests with")

if(NOT EXISTS "${LAMMPS_SOURCE_DIR}/min.h")
  message(FATAL_ERROR "LAMMPS_SOURCE_DIR must name the LAMMPS src directory (min.h not found)")
endif()
# "plugin load" registers min styles from the 22Jul2025 stable release on.
# 2Aug2023 and 29Aug2024 log "Loading plugins for min styles not yet
# implemented" and then reject min_style rgmin.
file(READ "${LAMMPS_SOURCE_DIR}/PLUGIN/plugin.cpp" lammps_plugin_cpp)
string(FIND "${lammps_plugin_cpp}" "pstyle == \"min\"" lammps_min_plugins)
if(lammps_min_plugins EQUAL -1)
  message(FATAL_ERROR "This LAMMPS cannot load min styles from plugins; "
    "min_style rgmin needs LAMMPS 22Jul2025 or later")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

get_filename_component(RGMIN_ROOT "${PROJECT_SOURCE_DIR}/.." ABSOLUTE)

if(RGMIN_LIBRARY)
  set(rgmin_lib "${RGMIN_LIBRARY}")
  add_custom_target(rgmin_cargo)
else()
  find_program(CARGO cargo REQUIRED)
  set(rgmin_target_dir "${CMAKE_BINARY_DIR}/cargo")
  set(rgmin_lib "${rgmin_target_dir}/release/${CMAKE_STATIC_LIBRARY_PREFIX}rgmin${CMAKE_STATIC_LIBRARY_SUFFIX}")
  add_custom_target(rgmin_cargo
    COMMAND ${CMAKE_COMMAND} -E env CARGO_TARGET_DIR=${rgmin_target_dir}
            ${CARGO} build --release --locked --lib --features capi
    WORKING_DIRECTORY "${RGMIN_ROOT}"
    BYPRODUCTS "${rgmin_lib}"
    COMMENT "Building rgmin (capi) with cargo"
    USES_TERMINAL)
endif()

find_package(Threads REQUIRED)

add_library(rgminplugin MODULE src/rgminplugin.cpp src/min_rgmin.cpp)
add_dependencies(rgminplugin rgmin_cargo)
target_include_directories(rgminplugin PRIVATE "${LAMMPS_SOURCE_DIR}" "${RGMIN_ROOT}/include")
string(TOUPPER "${LAMMPS_SIZES}" lammps_sizes_upper)
target_compile_definitions(rgminplugin PRIVATE LAMMPS_${lammps_sizes_upper})

if(LAMMPS_MPI)
  set(MPI_CXX_SKIP_MPICXX TRUE)
  find_package(MPI REQUIRED COMPONENTS CXX)
  target_link_libraries(rgminplugin PRIVATE MPI::MPI_CXX)
else()
  target_include_directories(rgminplugin PRIVATE "${LAMMPS_SOURCE_DIR}/STUBS")
endif()

target_link_libraries(rgminplugin PRIVATE "${rgmin_lib}" Threads::Threads ${CMAKE_DL_LIBS})
if(NOT APPLE AND NOT WIN32)
  target_link_libraries(rgminplugin PRIVATE m)
endif()

set_target_properties(rgminplugin PROPERTIES PREFIX "" SUFFIX ".so")
if(APPLE)
  # LAMMPS symbols resolve against the lmp binary that loads the plugin.
  set_target_properties(rgminplugin PROPERTIES LINK_FLAGS "-Wl,-undefined,dynamic_lookup")
elseif(UNIX)
  # Keep the linked-in rgmin symbols private to the plugin.
  set_target_properties(rgminplugin PROPERTIES LINK_FLAGS "-Wl,--exclude-libs,ALL")
endif()

include(GNUInstallDirs)
install(TARGETS rgminplugin LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/lammps/plugins")

enable_testing()
if(RGMIN_LAMMPS_LMP)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_test(NAME min_rgmin
    COMMAND ${CMAKE_COMMAND} -E env
            RGMIN_LAMMPS_LMP=${RGMIN_LAMMPS_LMP}
            RGMIN_LAMMPS_PLUGIN=$<TARGET_FILE:rgminplugin>
            ${Python3_EXECUTABLE} -m pytest -q ${PROJECT_SOURCE_DIR}/tests)
endif()
