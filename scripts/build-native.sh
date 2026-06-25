#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
LLAMA_DIR=${LLAMA_DIR:-"$HOME/llama.cpp"}
BUILD_DIR=${BUILD_DIR:-"$LLAMA_DIR/build"}
INSTALL_DIR=${INSTALL_DIR:-"$HOME/pharo-infer-native"}
UNAME_S=$(uname -s)

case "$UNAME_S" in
  Darwin)
    RPATH_VALUE='@loader_path'
    ;;
  Linux)
    RPATH_VALUE='$ORIGIN'
    ;;
  *)
    echo "Unsupported platform for this helper script. Build native/ai_llama_shim.c as a shared library linked to llama.dll." >&2
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

mkdir -p "$INSTALL_DIR/lib"

LIBLLAMA=$(find "$BUILD_DIR" \( -name 'libllama.dylib' -o -name 'libllama.*.dylib' -o -name 'libllama.so' -o -name 'libllama.so.*' -o -name 'llama.dll' \) | head -n 1)
if [ -z "$LIBLLAMA" ]; then
  echo "Could not find libllama in $BUILD_DIR" >&2
  exit 1
fi

LIB_DIR=$(dirname "$LIBLLAMA")
INCLUDE_FLAGS="-I$LLAMA_DIR/include -I$LLAMA_DIR/ggml/include"

case "$UNAME_S" in
  Darwin)
    cc -dynamiclib -fPIC $INCLUDE_FLAGS "$ROOT_DIR/native/ai_llama_shim.c" \
      -L"$LIB_DIR" -lllama -Wl,-rpath,"$RPATH_VALUE" \
      -o "$INSTALL_DIR/lib/libai_llama.dylib"
    find "$LIB_DIR" -maxdepth 1 -name '*.dylib' -exec cp -P {} "$INSTALL_DIR/lib/" \;
    ;;
  Linux)
    cc -shared -fPIC $INCLUDE_FLAGS "$ROOT_DIR/native/ai_llama_shim.c" \
      -L"$LIB_DIR" -lllama -Wl,-rpath,"$RPATH_VALUE" \
      -o "$INSTALL_DIR/lib/libai_llama.so"
    find "$LIB_DIR" -maxdepth 1 \( -name '*.so' -o -name '*.so.*' \) -exec cp -P {} "$INSTALL_DIR/lib/" \;
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
esac
