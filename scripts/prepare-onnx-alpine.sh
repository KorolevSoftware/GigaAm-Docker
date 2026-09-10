#!/bin/sh
# Build once per architecture; ordinary service builds reuse the exported library.
set -eu
cd "$(dirname "$0")/.."

arch=${1:-$(docker version --format '{{.Server.Arch}}')}
case "$arch" in
  amd64|arm64) ;;
  *) echo "Unsupported architecture: $arch (expected amd64 or arm64)" >&2; exit 1 ;;
esac
jobs=${ORT_BUILD_JOBS:-$(docker info --format '{{.NCPU}}')}
case "$jobs" in
  ''|*[!0-9]*|0) echo 'ORT_BUILD_JOBS must be a positive integer' >&2; exit 1 ;;
esac

mkdir -p .onnxruntime
staging=$(mktemp -d ".onnxruntime/.${arch}.XXXXXX")
trap 'rm -rf "$staging"' EXIT
trap 'exit 1' HUP INT TERM

printf 'Preparing ONNX Runtime 1.29.0 for linux/%s using %s build jobs\n' "$arch" "$jobs"
docker buildx build --progress=plain --platform "linux/$arch" \
  -f Dockerfile.onnx-alpine --target artifact \
  --build-arg "ORT_BUILD_JOBS=$jobs" \
  --output "type=local,dest=$staging" .
chmod 755 "$staging"

# Keep the last working bundle if compilation fails; retain one backup on success.
destination=".onnxruntime/$arch"
if [ -d "$destination" ]; then
  rm -rf "$destination.previous"
  mv "$destination" "$destination.previous"
fi
mv "$staging" "$destination"
printf 'Prepared %s; keep this directory when cleaning the Docker build cache.\n' "$destination"
