#!/usr/bin/env bash
# Download the scikit-learn User Guide sources (.rst) at a pinned release.
set -euo pipefail

SKLEARN_VERSION="1.9.1"
DEST="data/raw/scikit-learn"

if [ -d "$DEST" ]; then
    echo "$DEST already exists, skipping download"
    exit 0
fi

git -c advice.detachedHead=false clone --quiet --depth 1 --branch "$SKLEARN_VERSION" \
  --filter=blob:none --sparse \
  https://github.com/scikit-learn/scikit-learn.git "$DEST"
git -C "$DEST" sparse-checkout set --no-cone '/doc/modules/*.rst'

echo "Downloaded $(ls "$DEST"/doc/modules/*.rst | wc -l | tr -d ' ') files to $DEST/doc/modules"
