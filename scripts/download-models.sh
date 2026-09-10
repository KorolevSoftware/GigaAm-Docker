#!/bin/sh
# Build-time model download. Requires curl, jq, sha256sum and wc.
set -eu
manifest=${1:?Manifest path required}
model_root=${2:?Destination required}
model=${3:-gigaam-v3-e2e-rnnt}
precision=${4:-fp32}

jq -e --arg model "$model" --arg precision "$precision" '
  any(.bundles[]; .id == $model and .precision == $precision) and
  any(.bundles[]; .id == "silero-vad" and .precision == "fp32")
' "$manifest" > /dev/null

files=$(mktemp)
trap 'rm -f "$files"' EXIT
jq -r --arg model "$model" --arg precision "$precision" '
  .bundles[] | select((.id == $model and .precision == $precision) or (.id == "silero-vad" and .precision == "fp32")) |
  (.id + "/" + .revision + "/" + .precision) as $dir |
  .files[] | [$dir + "/" + .name, .url, .size, .sha256] | @tsv
' "$manifest" > "$files"

while IFS="$(printf '\t')" read -r relative url size sha; do
  path="$model_root/$relative"
  mkdir -p "$(dirname "$path")"
  curl --fail --location --proto '=https' --proto-redir '=https' \
    --retry 3 --max-time 1800 --output "$path.part" "$url"
  test "$(wc -c < "$path.part" | tr -d '[:space:]')" = "$size"
  printf '%s  %s\n' "$sha" "$path.part" | sha256sum -c -
  mv "$path.part" "$path"
  chmod 444 "$path"
done < "$files"
