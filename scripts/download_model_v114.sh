#!/usr/bin/env bash
set -euo pipefail

REPO="seonananana/emotion_analyzer"
TAG="model-v114"
ASSET="koelectra_oov_checkpoint-114.tar.gz"

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "[x] GITHUB_TOKEN is not set."
  echo "    export GITHUB_TOKEN='github_pat_...'"
  exit 1
fi

echo "[+] Fetching release info via GitHub API..."
release_json="$(curl -fsSL \
  -H "Authorization: Bearer ${GITHUB_TOKEN}" \
  -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/${REPO}/releases/tags/${TAG}")"

asset_id="$(python3 -c '
import json,sys
d=json.loads(sys.argv[1])
name=sys.argv[2]
for a in d.get("assets", []):
    if a.get("name") == name:
        print(a.get("id"))
        sys.exit(0)
print("")
' "$release_json" "$ASSET")"

if [[ -z "$asset_id" ]]; then
  echo "[x] Asset not found in release tag ${TAG}: ${ASSET}"
  echo "    Available assets:"
  python3 -c 'import json,sys; d=json.loads(sys.argv[1]); print([a.get("name") for a in d.get("assets",[])])' "$release_json"
  exit 1
fi

echo "[+] Downloading asset id=${asset_id} ..."
mkdir -p models

curl -fL \
  -H "Authorization: Bearer ${GITHUB_TOKEN}" \
  -H "Accept: application/octet-stream" \
  -o "/tmp/${ASSET}" \
  "https://api.github.com/repos/${REPO}/releases/assets/${asset_id}"

echo "[+] Extracting..."
tar -xzf "/tmp/${ASSET}" -C .

if [[ ! -d "models/koelectra_oov/checkpoint-114" ]]; then
  echo "[x] Extracted, but expected path not found: models/koelectra_oov/checkpoint-114"
  echo "    Archive structure may be different."
  exit 1
fi

echo "[✓] Model ready: models/koelectra_oov/checkpoint-114"
