# Video2X server environment. Source it in every new shell before using the tools:
#
#     . ~/v2x/env.sh
#
# Everything lives under $V2X_HOME (default ~/v2x): the package manager, the build tools and
# libraries, the build, the installed program and every cache. Nothing is installed system-wide,
# and `scripts/server/cleanup.sh` (or `rm -rf ~/v2x`) removes it all.

export V2X_HOME="${V2X_HOME:-$HOME/v2x}"

# Keep the package manager and every cache inside $V2X_HOME
export MAMBA_ROOT_PREFIX="$V2X_HOME/mamba"
export CONDA_PKGS_DIRS="$V2X_HOME/mamba/pkgs"
export XDG_CACHE_HOME="$V2X_HOME/cache"
export PIP_CACHE_DIR="$V2X_HOME/cache/pip"
export __GL_SHADER_DISK_CACHE_PATH="$V2X_HOME/cache/nvidia"

# The installed program first, then the private tools (ffmpeg, python, cmake)
export PATH="$V2X_HOME/app/bin:$V2X_HOME/env/bin:$V2X_HOME/bin:$PATH"

# Use the host's NVIDIA Vulkan driver
if [ -z "${VK_ICD_FILENAMES:-}" ]; then
    for icd in /etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json; do
        if [ -f "$icd" ]; then
            export VK_ICD_FILENAMES="$icd"
            break
        fi
    done
fi
