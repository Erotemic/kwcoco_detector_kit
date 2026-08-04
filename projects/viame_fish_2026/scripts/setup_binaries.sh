#!/usr/bin/env bash
# Install one versioned VIAME binary archive and update viame-current.
#
# Usage:
#   bash setup_binaries.sh /path/to/VIAME-v0.22.7-Linux-64Bit.tar.gz

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

ARCHIVE="${1:-$VF_VIAME_ARCHIVE}"
VERSION="${2:-$VF_VIAME_VERSION}"

if [ ! -f "$ARCHIVE" ]; then
    echo "ERROR: VIAME archive does not exist: $ARCHIVE"
    echo "Copy it to aiq-gpu or pass its path as the first argument."
    exit 1
fi

mkdir -p "$VF_SOFTWARE_DPATH" "$VF_DOWNLOAD_DPATH"

ARCHIVE_STEM="$(basename "$ARCHIVE")"
ARCHIVE_STEM="${ARCHIVE_STEM%.tar.gz}"
ARCHIVE_STEM="${ARCHIVE_STEM%.tgz}"
ARCHIVE_STEM="${ARCHIVE_STEM%.tar}"
INSTALL_ROOT="$VF_SOFTWARE_DPATH/$ARCHIVE_STEM"

echo "Checking archive: $ARCHIVE"
tar -tf "$ARCHIVE" >/dev/null

if [ -e "$INSTALL_ROOT" ]; then
    echo "ERROR: versioned install directory already exists: $INSTALL_ROOT"
    echo "Remove it manually if you intentionally want to reinstall this version."
    exit 1
fi

mkdir -p "$INSTALL_ROOT"
tar -xf "$ARCHIVE" -C "$INSTALL_ROOT"

SETUP_FPATH="$(find "$INSTALL_ROOT" -type f -name setup_viame.sh -print -quit)"
if [ -z "$SETUP_FPATH" ]; then
    echo "ERROR: setup_viame.sh was not found after extraction"
    exit 1
fi

VIAME_INSTALL="$(dirname "$SETUP_FPATH")"
sha256sum "$ARCHIVE" > "$INSTALL_ROOT/archive.sha256"
cp "$INSTALL_ROOT/archive.sha256" "$VIAME_INSTALL/.viame_archive.sha256"

cat > "$INSTALL_ROOT/install_info.txt" <<INFO
version=$VERSION
archive=$ARCHIVE
archive_name=$(basename "$ARCHIVE")
installed_at=$(date --iso-8601=seconds)
viame_install=$VIAME_INSTALL
INFO
cp "$INSTALL_ROOT/install_info.txt" "$VIAME_INSTALL/.viame_install_info.txt"

rm -f "$VF_CURRENT_VIAME_LINK"
ln -s "$VIAME_INSTALL" "$VF_CURRENT_VIAME_LINK"

echo
echo "VIAME setup finished"
echo "Versioned root: $INSTALL_ROOT"
echo "VIAME install:  $VIAME_INSTALL"
echo "Current link:   $VF_CURRENT_VIAME_LINK"
ls -ld "$VF_CURRENT_VIAME_LINK"
