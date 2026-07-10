#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
LLAMA_DIR=${LLAMA_DIR:-"$HOME/llama.cpp"}
BUILD_DIR=${BUILD_DIR:-"$LLAMA_DIR/build"}
INSTALL_DIR=${INSTALL_DIR:-"$HOME/pharo-infer-native"}
SHIM_BUILD_DIR=${SHIM_BUILD_DIR:-"$INSTALL_DIR/build-shim"}
UNAME_S=$(uname -s)

case "$UNAME_S" in
  Darwin)
    RPATH_VALUE='@loader_path'
    SHIM_NAME='libai_llama.dylib'
    ;;
  Linux)
    RPATH_VALUE='$ORIGIN'
    SHIM_NAME='libai_llama.so'
    ;;
  MINGW*|MSYS*|CYGWIN*)
    RPATH_VALUE=''
    SHIM_NAME='ai_llama.dll'
    ;;
  *)
    echo "Unsupported platform for this helper script: $UNAME_S" >&2
    exit 1
    ;;
esac

if [ ! -d "$LLAMA_DIR" ]; then
  git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
fi

if [ ! -f "$LLAMA_DIR/include/llama.h" ]; then
  echo "Could not find llama.cpp headers under $LLAMA_DIR. Set LLAMA_DIR to a llama.cpp source directory." >&2
  exit 1
fi

cmake -S "$LLAMA_DIR" -B "$BUILD_DIR" \
  -DBUILD_SHARED_LIBS=ON \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_TOOLS=OFF \
  -DLLAMA_BUILD_APP=OFF \
  -DLLAMA_BUILD_COMMON=OFF \
  -DCMAKE_BUILD_RPATH="$RPATH_VALUE" \
  -DCMAKE_INSTALL_RPATH="$RPATH_VALUE" \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON

cmake --build "$BUILD_DIR" --config Release --target llama

LIBLLAMA=$(find "$BUILD_DIR" \( -name 'libllama.dylib' -o -name 'libllama.*.dylib' -o -name 'libllama.so' -o -name 'libllama.so.*' -o -name 'llama.dll' -o -name 'libllama.dll' -o -name 'llama.lib' -o -name 'libllama.dll.a' \) | head -n 1)
if [ -z "$LIBLLAMA" ]; then
  echo "Could not find libllama in $BUILD_DIR" >&2
  exit 1
fi

LIB_DIR=$(dirname "$LIBLLAMA")

cmake -S "$ROOT_DIR/native" -B "$SHIM_BUILD_DIR" \
  -DLLAMA_DIR="$LLAMA_DIR" \
  -DLLAMA_BUILD_DIR="$BUILD_DIR" \
  -DAI_LLAMA_RPATH="$RPATH_VALUE" \
  -DCMAKE_BUILD_TYPE=Release

cmake --build "$SHIM_BUILD_DIR" --config Release --target ai_llama

mkdir -p "$INSTALL_DIR/lib"

SHIM_LIB=$(find "$SHIM_BUILD_DIR" \( -name 'libai_llama.dylib' -o -name 'libai_llama.so' -o -name 'ai_llama.dll' \) | head -n 1)
if [ -z "$SHIM_LIB" ]; then
  echo "Could not find $SHIM_NAME in $SHIM_BUILD_DIR" >&2
  exit 1
fi

cp -f "$SHIM_LIB" "$INSTALL_DIR/lib/$SHIM_NAME"

case "$UNAME_S" in
  Darwin)
    find "$LIB_DIR" -maxdepth 1 -name '*.dylib' -exec cp -P {} "$INSTALL_DIR/lib/" \;
    ;;
  Linux)
    find "$LIB_DIR" -maxdepth 1 \( -name '*.so' -o -name '*.so.*' \) -exec cp -P {} "$INSTALL_DIR/lib/" \;
    ;;
  MINGW*|MSYS*|CYGWIN*)
    find "$BUILD_DIR" -type f -name '*.dll' -exec cp -f {} "$INSTALL_DIR/lib/" \;
    ;;
esac

echo "Native library installed in: $INSTALL_DIR/lib"
echo "Use this in Pharo:"
case "$UNAME_S" in
  Darwin)
    echo "AILlamaLibrary libraryPath: '$INSTALL_DIR/lib/libai_llama.dylib'."
    ;;
  Linux)
    echo "AILlamaLibrary libraryPath: '$INSTALL_DIR/lib/libai_llama.so'."
    ;;
  MINGW*|MSYS*|CYGWIN*)
    echo "AILlamaLibrary libraryPath: '$INSTALL_DIR/lib/ai_llama.dll'."
    ;;
esac
