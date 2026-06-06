#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUTLASS_DIR="${NOVADIT_CUTLASS_PATH:-$ROOT_DIR/third_party/cutlass}"
CUTLASS_TAG="${NOVADIT_CUTLASS_TAG:-v3.5.1}"
CUTLASS_REPO="${NOVADIT_CUTLASS_REPO:-https://github.com/NVIDIA/cutlass.git}"

mkdir -p "$(dirname "$CUTLASS_DIR")"

if [ -d "$CUTLASS_DIR/.git" ]; then
  echo "Updating CUTLASS in $CUTLASS_DIR to $CUTLASS_TAG"
  git -C "$CUTLASS_DIR" fetch --depth 1 origin "refs/tags/$CUTLASS_TAG:refs/tags/$CUTLASS_TAG"
  git -C "$CUTLASS_DIR" checkout -f "$CUTLASS_TAG"
else
  echo "Cloning CUTLASS $CUTLASS_TAG into $CUTLASS_DIR"
  git clone --depth 1 --branch "$CUTLASS_TAG" "$CUTLASS_REPO" "$CUTLASS_DIR"
fi

echo "CUTLASS ready:"
echo "  root: $CUTLASS_DIR"
echo "  cutlass headers: $CUTLASS_DIR/include/cutlass"
echo "  cute headers:    $CUTLASS_DIR/include/cute"
