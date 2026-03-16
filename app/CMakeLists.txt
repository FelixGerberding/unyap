cmake_minimum_required(VERSION 3.22)
project(whisperjni)

set(CMAKE_CXX_STANDARD 17)

# 16 KB page-size support: mandatory on Google Play for apps targeting SDK 35+.
# Force 16 KB ELF segment alignment on every shared/module library we produce
# here, including the ggml targets added via FetchContent below (libggml.so,
# libggml-base.so, and the libggml-cpu-*.so MODULE variants). Setting these
# before any add_library()/add_subdirectory() makes all targets inherit them.
set(_ALIGN_16K "-Wl,-z,max-page-size=16384,-z,common-page-size=16384")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${_ALIGN_16K}")
set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${_ALIGN_16K}")

set(WHISPER_LIB_DIR ${CMAKE_SOURCE_DIR}/src/main/cpp/whisper.cpp)

# Build whisper source
set(SOURCE_FILES
    ${WHISPER_LIB_DIR}/src/whisper.cpp
    ${CMAKE_SOURCE_DIR}/src/main/cpp/jni_bridge.cpp
)

find_library(LOG_LIB log)

include(FetchContent)

add_library(whisperjni SHARED ${SOURCE_FILES})

# Read version from whisper.cpp's CMakeLists.txt
file(READ "${WHISPER_LIB_DIR}/CMakeLists.txt" MAIN_CMAKE_CONTENT)
string(REGEX MATCH "project\\(\"whisper\\.cpp\" VERSION ([0-9]+\\.[0-9]+\\.[0-9]+)\\)" VERSION_MATCH "${MAIN_CMAKE_CONTENT}")
if(CMAKE_MATCH_1)
    set(WHISPER_VERSION "${CMAKE_MATCH_1}")
else()
    set(WHISPER_VERSION "unknown")
endif()

# NOTE: do NOT define GGML_USE_CPU here. The CPU backend is loaded dynamically
# at runtime (GGML_BACKEND_DL below), not linked statically.
target_compile_definitions(whisperjni PRIVATE WHISPER_VERSION="${WHISPER_VERSION}")

# Always optimize native code, even in debug builds
target_compile_options(whisperjni PRIVATE -O3)
target_compile_options(whisperjni PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
target_compile_options(whisperjni PRIVATE -ffunction-sections -fdata-sections)
target_link_options(whisperjni PRIVATE -Wl,--gc-sections)
target_link_options(whisperjni PRIVATE -Wl,--exclude-libs,ALL)
target_link_options(whisperjni PRIVATE -flto)

# IMPORTANT: whisperjni (and the whisper.cpp core compiled into it) is built at
# the plain arm64-v8a baseline (ARMv8.0-A). We must NOT raise -march here: the
# CPU-feature-specific kernels (dotprod / fp16 / i8mm) live in ggml and are
# selected at runtime via GGML_CPU_ALL_VARIANTS below. Forcing a high -march
# baseline here would emit e.g. i8mm instructions unconditionally and crash
# (SIGILL) on devices that lack them (every Pixel before the Pixel 8).

# Force release-level optimization for native code in all build types
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O3")

# Build ggml from the whisper.cpp bundled copy.
#
# Instead of forcing a single high -march (which crashes on older CPUs), build
# every Android CPU variant (armv8.0 baseline .. armv9.2) as separate dynamic
# backends and let ggml pick the best one the device actually supports at
# runtime, via getauxval(HWCAP) scoring.
#
#   GGML_CPU_ALL_VARIANTS - build android_armv8.0 .. android_armv9.2 variants
#   GGML_BACKEND_DL        - build each variant as a dlopen-able libggml-cpu-*.so
#   BUILD_SHARED_LIBS      - required by GGML_BACKEND_DL (libggml/libggml-base.so)
#   GGML_NATIVE OFF        - must be off; -march=native conflicts with variants
set(BUILD_SHARED_LIBS      ON  CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL        ON  CACHE BOOL "" FORCE)
set(GGML_CPU_ALL_VARIANTS  ON  CACHE BOOL "" FORCE)
set(GGML_NATIVE            OFF CACHE BOOL "" FORCE)

FetchContent_Declare(ggml SOURCE_DIR ${WHISPER_LIB_DIR}/ggml)
FetchContent_MakeAvailable(ggml)

target_link_libraries(whisperjni ${LOG_LIB} android ggml)

target_include_directories(whisperjni PRIVATE
    ${WHISPER_LIB_DIR}
    ${WHISPER_LIB_DIR}/src
    ${WHISPER_LIB_DIR}/include
    ${WHISPER_LIB_DIR}/ggml/include
    ${WHISPER_LIB_DIR}/ggml/src
    ${WHISPER_LIB_DIR}/ggml/src/ggml-cpu
)
