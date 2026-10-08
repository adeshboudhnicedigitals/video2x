#!/usr/bin/env bash
# Builds this fork of Video2X into one folder, $V2X_HOME (default ~/v2x), without root and
# without installing anything system-wide.
#
# Usage (on the server):
#     git clone https://github.com/adeshboudhnicedigitals/video2x.git ~/v2x/src
#     bash ~/v2x/src/scripts/server/setup.sh
#     . ~/v2x/env.sh
#
# Layout:
#     ~/v2x/bin/micromamba   package manager (one static binary)
#     ~/v2x/env              build tools and libraries from conda-forge (cmake, gcc, ffmpeg, Vulkan, Boost)
#     ~/v2x/src              this repository (with the fork's ncnn-options patch applied)
#     ~/v2x/build            CMake build tree (can be deleted after the build)
#     ~/v2x/app              installed video2x, libvideo2x and models
#     ~/v2x/cache, mamba     caches
#     ~/v2x/work             jobs and benchmarks
# It needs the host NVIDIA driver with its Vulkan part (/etc/vulkan/icd.d/nvidia_icd.json).

set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export V2X_HOME="${V2X_HOME:-$HOME/v2x}"
mkdir -p "$V2X_HOME"/bin "$V2X_HOME"/cache "$V2X_HOME"/work
cp "$SRC/scripts/server/env.sh" "$V2X_HOME/env.sh"
# shellcheck source=/dev/null
. "$V2X_HOME/env.sh"

echo "== Video2X server setup into $V2X_HOME (source: $SRC)"
df -h "$V2X_HOME" | tail -1

# --- 1. Host GPU and Vulkan driver ---
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
if [ -z "${VK_ICD_FILENAMES:-}" ]; then
    echo "No NVIDIA Vulkan ICD file found in /etc/vulkan/icd.d or /usr/share/vulkan/icd.d." >&2
    echo "Video2X needs the Vulkan part of the NVIDIA driver; stop here and report this." >&2
    exit 1
fi
echo "Vulkan ICD: $VK_ICD_FILENAMES"

# --- 2. Package manager ---
MM="$V2X_HOME/bin/micromamba"
if [ ! -x "$MM" ]; then
    echo "== Downloading micromamba"
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$V2X_HOME" bin/micromamba
fi

# --- 3. Build tools and libraries (private environment) ---
# glslang provides glslangValidator, which the model wrappers use to compile their shaders
PACKAGES=(python=3.12 numpy pandas
          cmake ninja make pkg-config c-compiler cxx-compiler
          "ffmpeg=7.*=gpl*" libvulkan-loader libvulkan-headers glslang libboost-devel)
if [ ! -d "$V2X_HOME/env/conda-meta" ]; then
    echo "== Creating the private environment (a few minutes)"
    "$MM" create -y -p "$V2X_HOME/env" -c conda-forge --override-channels "${PACKAGES[@]}"
else
    echo "== Updating the private environment (installs anything missing)"
    "$MM" install -y -p "$V2X_HOME/env" -c conda-forge --override-channels "${PACKAGES[@]}"
fi

# --- 4. Submodules and the fork's patch ---
echo "== Fetching submodules"
git -C "$SRC" submodule update --init --recursive -- \
    third_party/ncnn third_party/spdlog \
    third_party/librealesrgan_ncnn_vulkan third_party/librealcugan_ncnn_vulkan third_party/librife_ncnn_vulkan

PATCH="$SRC/patches/librealesrgan-ncnn-options.patch"
WRAPPER="$SRC/third_party/librealesrgan_ncnn_vulkan"
if git -C "$WRAPPER" apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "Patch already applied: $(basename "$PATCH")"
else
    git -C "$WRAPPER" apply "$PATCH"
    echo "Applied $(basename "$PATCH")"
fi

# --- 5. Build and install into $V2X_HOME/app ---
echo "== Building (uses all cores)"
# A failed earlier configure can leave tool paths cached as not found
rm -f "$V2X_HOME/build/CMakeCache.txt"
# ncnn's bundled glslang is built as shared libraries. conda's linker is a cross linker, so it
# does not follow libvideo2x's rpath to find them when linking the video2x program.
GLSLANG_BUILD="$V2X_HOME/build/third_party/ncnn/glslang"
RPATH_LINK="-Wl,-rpath-link,$GLSLANG_BUILD/glslang:$GLSLANG_BUILD/SPIRV"
"$MM" run -p "$V2X_HOME/env" cmake -G Ninja -S "$SRC" -B "$V2X_HOME/build" \
    -DCMAKE_EXE_LINKER_FLAGS="$RPATH_LINK" \
    -DCMAKE_SHARED_LINKER_FLAGS="$RPATH_LINK" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$V2X_HOME/app" \
    -DCMAKE_PREFIX_PATH="$V2X_HOME/env" \
    -DVIDEO2X_USE_EXTERNAL_NCNN=OFF \
    -DVIDEO2X_USE_EXTERNAL_SPDLOG=OFF \
    -DVIDEO2X_USE_EXTERNAL_BOOST=ON \
    -DCMAKE_INSTALL_RPATH="$V2X_HOME/app/lib;$V2X_HOME/env/lib" \
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON
"$MM" run -p "$V2X_HOME/env" cmake --build "$V2X_HOME/build" --target install --parallel

# video2x looks for models next to its binary when they are not in /usr/share/video2x
ln -sfn ../share/video2x/models "$V2X_HOME/app/bin/models"

# --- 6. Check ---
echo "== Checking the build"
video2x --version
video2x --list-devices
TEST="$V2X_HOME/work/setup_check"
mkdir -p "$TEST"
ffmpeg -y -loglevel error -f lavfi -i testsrc2=size=320x180:rate=24 -t 1 -pix_fmt yuv420p "$TEST/in.mp4"
video2x -i "$TEST/in.mp4" -o "$TEST/out.mp4" -p realesrgan --realesrgan-model realesr-animevideov3 -s 2 \
    --log-level info --no-progress > "$TEST/log.txt" 2>&1 || true
grep -E "fp16-8x8x16|ncnn options|Total frames processed" "$TEST/log.txt" || true
frames="$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames,width,height \
    -of csv=p=0 "$TEST/out.mp4" 2>/dev/null || echo "none")"
echo "Test output (width,height,frames): $frames (expected 640,360,24)"
rm -rf "$TEST"

echo
echo "Done. Disk used by $V2X_HOME:"
du -sh "$V2X_HOME"
echo "The build tree can be deleted to save space: rm -rf $V2X_HOME/build"
echo "In every new shell: . $V2X_HOME/env.sh"
