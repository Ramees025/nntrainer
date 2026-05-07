#!/bin/bash

# Script to build the tokenizers-cpp static library for the host (desktop) target.
# Produces $SCRIPT_DIR/lib/libtokenizers_c.a, the same artifact consumed by
# Applications/CausalLM/meson.build.
#
# Mirrors build_tokenizer_android.sh but skips Android cross-compilation: the
# nntrainer patches are applied identically so binary tokenizer I/O symbols
# (tokenizers_new_from_bin / tokenizers_save_to_bin) are available.

set -e

echo "Building tokenizers-cpp library for host desktop..."

# Prerequisites
if ! command -v cmake >/dev/null 2>&1; then
    echo "Error: cmake is not installed. Please install cmake."
    exit 1
fi
if ! command -v rustc >/dev/null 2>&1 || ! command -v cargo >/dev/null 2>&1; then
    echo "Error: Rust is not installed. Please install Rust from https://rustup.rs/"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/tokenizers-cpp-build"

# Clone if needed
if [ ! -d "$BUILD_DIR/tokenizers-cpp" ]; then
    echo "Cloning tokenizers-cpp repository..."
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    git clone https://github.com/mlc-ai/tokenizers-cpp.git
fi

cd "$BUILD_DIR/tokenizers-cpp"
echo "Updating submodules..."
git submodule update --init --recursive

# Apply nntrainer patches (binary tokenizer I/O extension)
PATCH_SCRIPT="$SCRIPT_DIR/patches/apply_patches.sh"
PATCH_STAMP="$BUILD_DIR/tokenizers-cpp/.nntrainer_patch_stamp"
if [ -f "$PATCH_SCRIPT" ]; then
    PATCH_HASH=""
    if command -v sha256sum >/dev/null 2>&1; then
        PATCH_HASH=$(sha256sum "$SCRIPT_DIR/patches/bin_io_append.rs" "$PATCH_SCRIPT" 2>/dev/null | awk '{print $1}' | tr -d '\n')
    elif command -v shasum >/dev/null 2>&1; then
        PATCH_HASH=$(shasum -a 256 "$SCRIPT_DIR/patches/bin_io_append.rs" "$PATCH_SCRIPT" 2>/dev/null | awk '{print $1}' | tr -d '\n')
    fi
    if [ ! -f "$PATCH_STAMP" ] || [ "$(cat "$PATCH_STAMP" 2>/dev/null)" != "$PATCH_HASH" ]; then
        echo "Applying nntrainer patches..."
        bash "$PATCH_SCRIPT" "$BUILD_DIR/tokenizers-cpp"
        echo "$PATCH_HASH" > "$PATCH_STAMP"
        rm -rf "$BUILD_DIR/tokenizers-cpp/build-desktop"
    else
        echo "nntrainer patches already up to date."
    fi
else
    echo "Warning: nntrainer patch script not found at $PATCH_SCRIPT"
fi

mkdir -p "build-desktop"
cd "build-desktop"

echo "Configuring CMake for desktop..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DTOKENIZERS_CPP_BUILD_TESTS=OFF \
    -DTOKENIZERS_CPP_BUILD_EXAMPLES=OFF

echo "Building tokenizers-cpp..."
NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
cmake --build . -j"$NPROC"

mkdir -p "$SCRIPT_DIR/lib"

# Find the generated libtokenizers_c.a (single combined static lib).
LIB_PATH=$(find "$BUILD_DIR/tokenizers-cpp/build-desktop" -name "libtokenizers_c.a" -type f | head -n 1)
if [ -z "$LIB_PATH" ]; then
    # Fall back: combine separate libs into one.
    LIBS_TO_COMBINE=$(find "$BUILD_DIR/tokenizers-cpp/build-desktop" -name "*.a" -type f | grep -v "CMakeFiles" | tr '\n' ' ')
    if [ -z "$LIBS_TO_COMBINE" ]; then
        echo "Error: No static libraries produced by cmake build."
        exit 1
    fi
    TEMP_DIR="$BUILD_DIR/temp_objs_desktop"
    rm -rf "$TEMP_DIR"
    mkdir -p "$TEMP_DIR"
    cd "$TEMP_DIR"
    for lib in $LIBS_TO_COMBINE; do
        ar x "$lib" || true
    done
    ar rcs "$SCRIPT_DIR/lib/libtokenizers_c.a" *.o
    cd ..
    rm -rf "$TEMP_DIR"
else
    cp "$LIB_PATH" "$SCRIPT_DIR/lib/libtokenizers_c.a"
fi

if [ -f "$SCRIPT_DIR/lib/libtokenizers_c.a" ]; then
    echo "Build completed successfully!"
    echo "Library copied to: $SCRIPT_DIR/lib/libtokenizers_c.a"
else
    echo "Error: Failed to produce libtokenizers_c.a"
    exit 1
fi
