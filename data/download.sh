#!/bin/sh
# Fetch the pinned official code that the baselines and benchmarks use, into data/ext/ (gitignored).
# Each repo is checked out at a fixed commit; the script fails if the checked-out commit differs.
# Nothing here is a pretrained model or dataset: MQAR and the RULER-style tasks are generated
# by the official generators from a seed.
set -e
cd "$(dirname "$0")"
mkdir -p ext
fetch() {  # name url commit
  if [ ! -d "ext/$1/.git" ]; then git clone --quiet --filter=blob:none "$2" "ext/$1"; fi
  git -C "ext/$1" fetch --quiet origin "$3" 2>/dev/null || true
  git -C "ext/$1" checkout --quiet "$3"
  got=$(git -C "ext/$1" rev-parse HEAD)
  [ "$got" = "$3" ] || { echo "commit mismatch for $1: $got"; exit 1; }
  echo "ok $1 $3"
}
fetch flash-linear-attention https://github.com/fla-org/flash-linear-attention cc4648a3b35c27c6bc6b3cfb0079a14d57bcbc5b  # MIT
fetch mamba    https://github.com/state-spaces/mamba   e9594ce1c732d97440f0332fdc43170a2294dbfa  # Apache-2.0
fetch based    https://github.com/HazyResearch/based  32c1bd4e09c8f7f718b0ed3f2e0a333908297879  # Apache-2.0
fetch zoology  https://github.com/HazyResearch/zoology 1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb  # Apache-2.0
fetch RULER    https://github.com/NVIDIA/RULER         db43f92ff634bbce1c33d1fb15858aec4bba509a  # Apache-2.0
# Checksums of the files the environment imports, so a changed file is caught even at the same commit.
shasum -a 256 -c checksums.sha256
