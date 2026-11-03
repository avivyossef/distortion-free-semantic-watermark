#!/usr/bin/env bash
# Clone the three native watermark baselines at the commits used in the paper.
# They are not vendored: install their own requirements and check their licenses
# before use.  Usage: third_party/clone.sh <target-directory>
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <target-directory>" >&2
  exit 2
fi
target=$1
mkdir -p "$target"

git_safe() {  # git with repository hooks disabled
  git -c core.hooksPath=/dev/null "$@"
}

clone_at() {  # <url> <directory> <commit>
  local url=$1 dir=$target/$2 commit=$3
  if [ -d "$dir/.git" ]; then
    git_safe -C "$dir" fetch --quiet origin "$commit"
  else
    git_safe clone --quiet "$url" "$dir"
  fi
  git_safe -C "$dir" checkout --quiet --detach "$commit"
  test "$(git -C "$dir" rev-parse HEAD)" = "$commit"
  echo "$2 at $commit"
}

clone_at https://github.com/YuxinWenRick/tree-ring-watermark tree-ring-watermark 3015283d9cf82e90b628f02ad2121bd37408ca9a
clone_at https://github.com/bsmhmmlf/Gaussian-Shading Gaussian-Shading 09c678fadc7545acf7be12647ddf2a5e66f6a9dc
clone_at https://github.com/Kasraarabi/SEAL SEAL 92d31b31b93a6e373fe88a584c640beafb68c6fa
