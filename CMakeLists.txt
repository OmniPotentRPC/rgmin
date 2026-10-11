cmake_minimum_required(VERSION 3.15...3.30)
project(rgmin_python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

if(MSVC)
  add_compile_options(/fp:strict)
else()
  add_compile_options(-ffp-contract=off)
endif()

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module Development.SABIModule)
find_package(nanobind CONFIG REQUIRED)

# T2: nightly rustc, an explicit target triple, and a std rebuilt with
# the thread sanitizer. Rust_TOOLCHAIN and Rust_CARGO_TARGET are read
# when Corrosion is included, so they are set first. The workflow passes
# the dated nightly that matches the image clang.
option(RGMIN_TSAN "Build the Rust staticlib with -Zsanitizer=thread (T2)." OFF)
if(RGMIN_TSAN)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    set(_rgmin_tsan_target "aarch64-unknown-linux-gnu")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    set(_rgmin_tsan_target "x86_64-unknown-linux-gnu")
  else()
    message(FATAL_ERROR
      "RGMIN_TSAN has no target triple for ${CMAKE_SYSTEM_PROCESSOR}")
  endif()
  if(NOT Rust_TOOLCHAIN)
    set(Rust_TOOLCHAIN "nightly" CACHE STRING
      "Rustup toolchain used when RGMIN_TSAN is ON")
  endif()
  set(Rust_CARGO_TARGET "${_rgmin_tsan_target}" CACHE STRING
    "Explicit cargo target triple for the sanitizer staticlib" FORCE)
  message(STATUS "RGMIN_TSAN toolchain ${Rust_TOOLCHAIN} target ${_rgmin_tsan_target}")
endif()

include(FetchContent)
if(POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)
endif()
FetchContent_Declare(
  corrosion
  GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
  GIT_TAG v0.6.1
)
# Corrosion 0.6.1 names the cargo target directory from the crate path
# relative to its grandparent. When that grandparent is the filesystem
# root the name stays absolute and cargo is asked to write under `/`.
FetchContent_GetProperties(corrosion)
if(NOT corrosion_POPULATED)
  FetchContent_Populate(corrosion)
  set(_corrosion_cmake "${corrosion_SOURCE_DIR}/cmake/Corrosion.cmake")
  file(READ "${_corrosion_cmake}" _corrosion_text)
  set(_corrosion_old [=[string(REPLACE "${grandparent_path}/" "" cargo_folder_name "${parent_path}")]=])
  set(_corrosion_new [=[string(REPLACE "${grandparent_path}/" "" cargo_folder_name "${parent_path}")
    if(IS_ABSOLUTE "${cargo_folder_name}")
        cmake_path(GET parent_path FILENAME cargo_folder_name)
    endif()]=])
  string(FIND "${_corrosion_text}" "${_corrosion_old}" _corrosion_at)
  if(_corrosion_at EQUAL -1)
    message(FATAL_ERROR "Corrosion.cmake no longer has the target-dir path splice")
  endif()
  string(REPLACE "${_corrosion_old}" "${_corrosion_new}" _corrosion_text "${_corrosion_text}")
  file(WRITE "${_corrosion_cmake}" "${_corrosion_text}")
  add_subdirectory("${corrosion_SOURCE_DIR}" "${corrosion_BINARY_DIR}")
endif()

corrosion_import_crate(
  MANIFEST_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml"
  CRATES rgmin
  CRATE_TYPES staticlib
  FEATURES capi
  LOCKED
)

if(RGMIN_TSAN)
  corrosion_set_cargo_flags(rgmin "-Zbuild-std=std,panic_abort")
  corrosion_add_target_rustflags(rgmin "-Zsanitizer=thread")
  # py-tsan.yml counts __tsan_func_entry in librust_core_demo.a.
  # corrosion_import_crate exposes rgmin as an INTERFACE library, which
  # cannot take a POST_BUILD rule. cargo-build_rgmin is the custom
  # target that runs after Corrosion copies librgmin.a into the build
  # tree, so the copy below is that same instrumented archive.
  add_custom_command(
    TARGET cargo-build_rgmin POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy
            "${CMAKE_BINARY_DIR}/librgmin.a"
            "${CMAKE_BINARY_DIR}/librust_core_demo.a"
    VERBATIM)
endif()

# F1: FREE_THREADED sets Py_mod_gil = Py_MOD_GIL_NOT_USED on a
# free-threaded interpreter. On a GIL interpreter nanobind keeps
# STABLE_ABI and ignores FREE_THREADED. cp312-abi3 covers 3.12+ GIL.
# cp314t and cp315t are version-specific free-threaded wheels.
nanobind_add_module(_core FREE_THREADED STABLE_ABI NB_STATIC python/src/ext.cpp)
target_include_directories(_core PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(_core PRIVATE rgmin)

set(_core_stage "${CMAKE_CURRENT_BINARY_DIR}/rgmin")
file(MAKE_DIRECTORY "${_core_stage}")
file(WRITE "${_core_stage}/__init__.py" "")
set_target_properties(_core PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "$<1:${_core_stage}>"
  RUNTIME_OUTPUT_DIRECTORY "$<1:${_core_stage}>")

if(NOT MSVC)
  target_compile_options(_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

# F9: Windows free-threaded builds define Py_GIL_DISABLED when the
# Python headers did not.
if(WIN32 AND NB_FREE_THREADED)
  target_compile_definitions(_core PRIVATE Py_GIL_DISABLED=1)
endif()

# F9: hidden visibility is the default above. --exclude-libs,ALL drops
# the staticlib from the dynamic table. The version script (Linux) and
# the export list (macOS) leave only PyInit__core.
if(UNIX AND NOT APPLE)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/core_exports.map"
"{\n  global:\n    PyInit__core;\n  local:\n    *;\n};\n")
  target_link_options(_core PRIVATE
    "LINKER:--exclude-libs,ALL"
    "LINKER:--version-script,${CMAKE_CURRENT_BINARY_DIR}/core_exports.map")
endif()
if(APPLE)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/core_exports.txt" "_PyInit__core\n")
  target_link_options(_core PRIVATE
    "LINKER:-exported_symbols_list,${CMAKE_CURRENT_BINARY_DIR}/core_exports.txt")
endif()
if(MSVC)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/core_exports.def" "EXPORTS\n  PyInit__core\n")
  target_link_options(_core PRIVATE
    "/DEF:${CMAKE_CURRENT_BINARY_DIR}/core_exports.def")
  add_custom_command(
    TARGET _core POST_BUILD
    COMMAND "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/ci/limit_pe_exports.py"
            "$<TARGET_FILE:_core>"
            "PyInit__core"
    VERBATIM)
endif()

nanobind_add_stub(
  _core_stub
  MODULE rgmin._core
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/_core.pyi"
  PYTHON_PATH "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS _core
)

install(TARGETS _core
  LIBRARY DESTINATION rgmin
  RUNTIME DESTINATION rgmin)
