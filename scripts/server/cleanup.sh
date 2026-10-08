#!/usr/bin/env bash
# Removes everything setup.sh created: $V2X_HOME (default ~/v2x), including the build, the
# private environment, the caches and the work folder. Copy any results you want to keep first.

set -euo pipefail

V2X_HOME="${V2X_HOME:-$HOME/v2x}"
if [ ! -d "$V2X_HOME" ]; then
    echo "$V2X_HOME does not exist; nothing to remove."
    exit 0
fi

echo "This deletes $V2X_HOME:"
du -sh "$V2X_HOME"/* 2>/dev/null || true
du -sh "$V2X_HOME"
read -r -p "Delete it? [y/N] " answer
if [ "$answer" != "y" ] && [ "$answer" != "Y" ]; then
    echo "Nothing deleted."
    exit 0
fi
rm -rf "$V2X_HOME"
echo "Deleted $V2X_HOME."

# Report (do not delete) folders that tools sometimes create in the home directory when the
# environment file was not sourced. They may also belong to other software, so check them by hand.
for leftover in "$HOME/.mamba" "$HOME/.conda" "$HOME/.cache/mamba" "$HOME/.cache/pip" "$HOME/.nv" "$HOME/.cache/nvidia"; do
    if [ -e "$leftover" ]; then
        echo "Note: $leftover exists ($(du -sh "$leftover" 2>/dev/null | cut -f1)). It may be from this job or from other software."
    fi
done
