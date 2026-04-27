#!/bin/bash
# =============================================================================
# rundocker.sh — Convenience wrapper for running PyCAD in Docker
#
# Usage:
#   ./rundocker.sh --test 4              Run a built-in test
#   ./rundocker.sh --file my_test.yaml   Run a YAML test file
#   ./rundocker.sh --dir examples/       Run all YAMLs in a directory
#   ./rundocker.sh --quick               Quick check
#   ./rundocker.sh --benchmarks          Full benchmark suite
#   ./rundocker.sh --shell               Open an interactive bash shell
#
# Output files appear in ./outputs/
# =============================================================================
set -e

IMAGE="pycad"
OUTPUTS_DIR="$(pwd)/outputs"
mkdir -p "$OUTPUTS_DIR"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Start Docker and try again."
    exit 1
fi

# Check if image exists
if ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
    echo "Error: Docker image '$IMAGE' not found."
    echo "Either pull it:  docker pull ghcr.io/mhbn/pycad:latest && docker tag ghcr.io/mhbn/pycad:latest pycad"
    echo "Or build it:     docker build -t pycad ."
    exit 1
fi

# Handle special cases
case "$1" in
    --shell)
        echo "Opening interactive shell..."
        docker run --rm -it \
            -v "$OUTPUTS_DIR":/opt/pycad/outputs \
            --entrypoint bash \
            "$IMAGE"
        exit 0
        ;;
    --file)
        # Mount the YAML file into the container
        if [ -z "$2" ]; then
            echo "Error: --file requires a path to a YAML file."
            exit 1
        fi
        YAML_PATH="$(realpath "$2")"
        YAML_NAME="$(basename "$2")"
        shift 2
        docker run --rm \
            -v "$OUTPUTS_DIR":/opt/pycad/outputs \
            -v "$YAML_PATH":/opt/pycad/"$YAML_NAME" \
            "$IMAGE" --file "$YAML_NAME" "$@"
        exit 0
        ;;
    --dir)
        # Mount the directory into the container
        if [ -z "$2" ]; then
            echo "Error: --dir requires a path to a directory."
            exit 1
        fi
        DIR_PATH="$(realpath "$2")"
        DIR_NAME="$(basename "$2")"
        shift 2
        docker run --rm \
            -v "$OUTPUTS_DIR":/opt/pycad/outputs \
            -v "$DIR_PATH":/opt/pycad/"$DIR_NAME" \
            "$IMAGE" --dir "$DIR_NAME" "$@"
        exit 0
        ;;
    *)
        # Pass everything through (--test, --quick, --benchmarks, etc.)
        docker run --rm \
            -v "$OUTPUTS_DIR":/opt/pycad/outputs \
            "$IMAGE" "$@"
        ;;
esac