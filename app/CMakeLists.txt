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

# Compile a patched copy of whisper.cpp, leaving the submodule untouched.
#
# With language auto-detection, whisper_full encodes the first 30s window once
# to detect the language and then again for the main loop. The patch applies
# audio_ctx before detection (so both passes are identical) and lets the main
# loop reuse the detection pass instead of encoding the window a second time.
# Each anchor must match exactly, so a submodule update that moves the code
# fails the configure step instead of silently dropping the patch.
set(WHISPER_SRC ${WHISPER_LIB_DIR}/src/whisper.cpp)
set(WHISPER_SRC_PATCHED ${CMAKE_CURRENT_BINARY_DIR}/whisper_patched/whisper.cpp)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${WHISPER_SRC})
file(READ ${WHISPER_SRC} _whisper_src)

function(patch_whisper_src anchor replacement)
    string(FIND "${_whisper_src}" "${anchor}" _pos)
    if(_pos EQUAL -1)
        message(FATAL_ERROR "whisper.cpp patch anchor not found, update the patch in CMakeLists.txt:\n${anchor}")
    endif()
    string(REPLACE "${anchor}" "${replacement}" _patched "${_whisper_src}")
    set(_whisper_src "${_patched}" PARENT_SCOPE)
endfunction()

patch_whisper_src(
[=[    // auto-detect language if not specified
]=]
[=[    // [unyap] apply audio_ctx before language detection, so its encoder pass
    // matches the main loop's first window and can be reused there
    if (params.audio_ctx > whisper_n_audio_ctx(ctx)) {
        WHISPER_LOG_ERROR("%s: audio_ctx is larger than the maximum allowed (%d > %d)\n", __func__, params.audio_ctx, whisper_n_audio_ctx(ctx));
        return -5;
    }
    state->exp_n_audio_ctx = params.audio_ctx;
    bool unyap_seek0_encoded = false;

    // auto-detect language if not specified
]=])

patch_whisper_src(
[=[        params.language = whisper_lang_str(lang_id);
]=]
[=[        params.language = whisper_lang_str(lang_id);
        unyap_seek0_encoded = true; // [unyap]
]=])

patch_whisper_src(
[=[        if (!whisper_encode_internal(*ctx, *state, seek, params.n_threads, params.abort_callback, params.abort_callback_user_data)) {
]=]
[=[        // [unyap] the window at seek 0 was already encoded for language detection
        const bool unyap_reuse_encoding = unyap_seek0_encoded && seek == 0;
        unyap_seek0_encoded = false;
        if (!unyap_reuse_encoding && !whisper_encode_internal(*ctx, *state, seek, params.n_threads, params.abort_callback, params.abort_callback_user_data)) {
]=])

# Write via configure_file so the copy (and its object file) only changes when
# the patched source actually does.
file(WRITE ${WHISPER_SRC_PATCHED}.tmp "${_whisper_src}")
configure_file(${WHISPER_SRC_PATCHED}.tmp ${WHISPER_SRC_PATCHED} COPYONLY)

# Build whisper source
set(SOURCE_FILES
    ${WHISPER_SRC_PATCHED}
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

# Force release-level optimization for native code in all build types. The
# per-build-type flags come after CMAKE_C_FLAGS on the command line, so they
# must carry the level too: release builds use RelWithDebInfo, whose default
# -O2 would otherwise override it for all of ggml.
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
foreach(_cfg DEBUG RELWITHDEBINFO RELEASE)
    set(CMAKE_C_FLAGS_${_cfg} "${CMAKE_C_FLAGS_${_cfg}} -O3")
    set(CMAKE_CXX_FLAGS_${_cfg} "${CMAKE_CXX_FLAGS_${_cfg}} -O3")
endforeach()

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

# Decode Opus (WhatsApp/Telegram voice notes) in-process with libopus (git
# submodule, v1.5.2) instead of MediaCodec, whose per-packet IPC costs ~1 ms per
# 20 ms packet: a minute of audio took 2.5-7 s through MediaCodec, 0.08 s here.
set(OPUS_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(OPUS_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
set(OPUS_INSTALL_PKG_CONFIG_MODULE OFF CACHE BOOL "" FORCE)
set(OPUS_INSTALL_CMAKE_CONFIG_MODULE OFF CACHE BOOL "" FORCE)
# ggml forces shared libraries; link libopus statically into whisperjni.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_SHARED_LIBS OFF)
add_subdirectory(${CMAKE_SOURCE_DIR}/src/main/cpp/opus EXCLUDE_FROM_ALL)
unset(BUILD_SHARED_LIBS)
target_link_libraries(whisperjni opus)

target_link_libraries(whisperjni ${LOG_LIB} android ggml)

target_include_directories(whisperjni PRIVATE
    ${WHISPER_LIB_DIR}
    ${WHISPER_LIB_DIR}/src
    ${WHISPER_LIB_DIR}/include
    ${WHISPER_LIB_DIR}/ggml/include
    ${WHISPER_LIB_DIR}/ggml/src
    ${WHISPER_LIB_DIR}/ggml/src/ggml-cpu
)
