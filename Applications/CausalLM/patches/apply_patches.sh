#!/bin/bash
# Idempotent patcher for mlc-ai/tokenizers-cpp Rust crate.
#
# Adds two FFI symbols to the Rust crate so the C++ side can load and persist
# tokenizers in a fast binary format (MessagePack) instead of JSON:
#   - tokenizers_new_from_bin
#   - tokenizers_save_to_bin
#
# Usage:
#   apply_patches.sh <path-to-cloned-tokenizers-cpp>
#
# Safe to run multiple times; markers are used to detect a previous run.

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <path-to-cloned-tokenizers-cpp>" >&2
    exit 1
fi

REPO_DIR="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LIB_RS="$REPO_DIR/rust/src/lib.rs"
CARGO_TOML="$REPO_DIR/rust/Cargo.toml"
APPEND_SRC="$SCRIPT_DIR/bin_io_append.rs"
MARKER="NNTRAINER_BIN_IO_BEGIN"

if [ ! -f "$LIB_RS" ]; then
    echo "[apply_patches] Error: $LIB_RS not found." >&2
    exit 1
fi

if [ ! -f "$CARGO_TOML" ]; then
    echo "[apply_patches] Error: $CARGO_TOML not found." >&2
    exit 1
fi

if [ ! -f "$APPEND_SRC" ]; then
    echo "[apply_patches] Error: $APPEND_SRC not found." >&2
    exit 1
fi

# 1) Append bin_io extension to lib.rs (idempotent)
if grep -q "$MARKER" "$LIB_RS"; then
    echo "[apply_patches] lib.rs already patched, skipping append."
else
    echo "[apply_patches] Appending bin_io extension to lib.rs..."
    printf '\n' >> "$LIB_RS"
    cat "$APPEND_SRC" >> "$LIB_RS"
fi

# 2) Add rmp-serde dependency to Cargo.toml (idempotent)
if grep -q '^rmp-serde' "$CARGO_TOML"; then
    echo "[apply_patches] rmp-serde already in Cargo.toml, skipping."
else
    echo "[apply_patches] Adding rmp-serde to Cargo.toml..."
    # Insert under [dependencies]; if the section is at end-of-file this still appends.
    awk '
        BEGIN { added = 0 }
        {
            print
            if (added == 0 && $0 ~ /^\[dependencies\]/) {
                print ""
                print "rmp-serde = \"1.3\""
                added = 1
            }
        }
        END {
            if (added == 0) {
                print ""
                print "[dependencies]"
                print "rmp-serde = \"1.3\""
            }
        }
    ' "$CARGO_TOML" > "$CARGO_TOML.tmp"
    mv "$CARGO_TOML.tmp" "$CARGO_TOML"
fi

echo "[apply_patches] Done."
