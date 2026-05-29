#!/usr/bin/env bash

# build_cuda.sh
# Recursively finds all .cu files in subdirectories of CWD,
# compiles each with nvcc, and places binaries in ./build/

set -uo pipefail  # removed -e so a single failure doesn't abort the whole run

ROOT_DIR="$(pwd)"
BUILD_DIR="$ROOT_DIR/build"
SUCCESS=0
FAIL=0
FAILED_FILES=()

# Create build output directory
mkdir -p "$BUILD_DIR"

echo "=== CUDA Build Script ==="
echo "Root:  $ROOT_DIR"
echo "Build: $BUILD_DIR"
echo ""

# Find all .cu files recursively
while IFS= read -r -d '' cu_file; do
    # Derive a unique output name using the relative path:
    # e.g.  ./subdir/kernels/vec_add.cu  ->  subdir_kernels_vec_add
    rel_path="${cu_file#"$ROOT_DIR"/}"          # strip leading root path
    exec_name="${rel_path%.cu}"                 # remove .cu extension
    exec_name="${exec_name//\//_}"              # replace / with _

    output_exec="$BUILD_DIR/$exec_name"

    echo "Compiling: $rel_path"
    echo "  -> $output_exec"

    # Detect Dynamic Parallelism files by _rdc suffix → need -rdc=true -lcudadevrt
    nvcc_flags=""
    if [[ "$cu_file" == *_rdc.cu ]]; then
        nvcc_flags="-rdc=true -lcudadevrt"
        echo "  (dynamic parallelism: $nvcc_flags)"
    fi

    # Use || true so a failed nvcc never triggers pipefail
    # shellcheck disable=SC2086
    if nvcc $nvcc_flags "$cu_file" -o "$output_exec" 2>&1; then
        echo "  [OK]"
        ((SUCCESS++)) || true
    else
        echo "  [FAILED]"
        FAILED_FILES+=("$rel_path")
        ((FAIL++)) || true
    fi

    echo ""
done < <(find "$ROOT_DIR" -mindepth 2 -name "*.cu" -print0)

echo "=== Build Complete ==="
echo "  Succeeded: $SUCCESS"
echo "  Failed:    $FAIL"

if (( FAIL > 0 )); then
    echo ""
    echo "Failed files:"
    for f in "${FAILED_FILES[@]}"; do
        echo "  - $f"
    done
    exit 1
fi