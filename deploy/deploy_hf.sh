#!/bin/bash
# Deploy to Hugging Face Spaces.
#
# Run this yourself — it pushes to your account and makes the demo public.
#
#   ./deploy/deploy_hf.sh <your-hf-username> [space-name]
#
# Prerequisites:
#   pip install huggingface_hub          # provides the `huggingface-cli` command
#   huggingface-cli login                # once, stores your token
#   git lfs install
#
# What it does:
#   1. verifies the artifacts and model weights the app needs are present
#   2. creates the Space if it does not exist (Docker SDK, free CPU tier)
#   3. pushes the repo with the Space README frontmatter in place
set -euo pipefail

USERNAME="${1:-}"
SPACE_NAME="${2:-reverse-turing-test}"

if [ -z "$USERNAME" ]; then
    echo "usage: $0 <your-hf-username> [space-name]" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- 1. Preflight -----------------------------------------------------------
echo "==> Checking required files"
MISSING=0
for f in artifacts/metadata.json artifacts/explorer.json artifacts/pca.pkl \
         models/meta_classifier.pkl models/stylometric_classifier.pkl \
         models/cnn_classifier.pt models/bert_classifier/model.safetensors; do
    if [ ! -f "$f" ]; then
        echo "    MISSING: $f" >&2
        MISSING=1
    fi
done
if [ "$MISSING" -ne 0 ]; then
    echo "" >&2
    echo "Run the build step first:  python src/build_artifacts.py --dataset raid" >&2
    exit 1
fi
echo "    all present"

command -v huggingface-cli >/dev/null || {
    echo "huggingface-cli not found. pip install huggingface_hub" >&2; exit 1; }
command -v git-lfs >/dev/null || {
    echo "git-lfs not found. brew install git-lfs && git lfs install" >&2; exit 1; }

SPACE_ID="$USERNAME/$SPACE_NAME"
SPACE_URL="https://huggingface.co/spaces/$SPACE_ID"

# --- 2. Verify the token can actually write ---------------------------------
# A read-scoped token fails here rather than three steps later at `git push`,
# where the error surfaces as a confusing "could not clone".
echo "==> Checking credentials"
python3 - "$SPACE_ID" <<'PY' || exit 1
import sys
from huggingface_hub import HfApi

api = HfApi()
try:
    me = api.whoami()
except Exception as exc:
    sys.exit(f"    Not logged in ({exc.__class__.__name__}). Run: hf auth login")

role = (me.get("auth") or {}).get("accessToken", {}).get("role")
print(f"    user: {me.get('name')}  token role: {role or 'unknown'}")
if role == "read":
    sys.exit(
        "    This token is READ-ONLY and cannot create or push to a Space.\n"
        "    Create a write token at https://huggingface.co/settings/tokens\n"
        "    then run: hf auth login"
    )

space_id = sys.argv[1]
try:
    api.space_info(space_id)
    print(f"    Space exists: {space_id}")
except Exception:
    print(f"    Creating Space: {space_id}")
    api.create_repo(space_id, repo_type="space", space_sdk="docker", exist_ok=True)
    print("    created")
PY

# --- 3. Stage and push ------------------------------------------------------
STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT
echo "==> Staging in $STAGING"

git clone "$SPACE_URL" "$STAGING/space" 2>/dev/null || {
    echo "Could not clone the Space. Is huggingface-cli logged in?" >&2; exit 1; }

cd "$STAGING/space"
git lfs install --local

# Space README carries the required YAML frontmatter.
cp "$REPO_ROOT/deploy/README-space.md" README.md
cp "$REPO_ROOT/Dockerfile" "$REPO_ROOT/.dockerignore" "$REPO_ROOT/requirements.txt" .
cp "$REPO_ROOT/.gitattributes" .
mkdir -p src app artifacts models
cp "$REPO_ROOT/src/features.py" "$REPO_ROOT/src/model_defs.py" src/
cp -R "$REPO_ROOT/app/." app/
cp -R "$REPO_ROOT/artifacts/." artifacts/
cp -R "$REPO_ROOT/models/." models/
find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
find . -name '.DS_Store' -delete 2>/dev/null || true

git add -A
if git diff --cached --quiet; then
    echo "==> No changes to push"
else
    git commit -m "Deploy Reverse Turing Test"
    echo "==> Pushing (~280 MB of weights on first push, this takes a while)"
    git push
fi

echo ""
echo "Done: $SPACE_URL"
echo "The Space will build for a few minutes before it goes live."
