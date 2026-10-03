#!/usr/bin/env bash
# Fix loop: benchmark BEFORE_REF and AFTER_REF from clean worktrees on the same data and model-output cache,
# then write a before/after gate table and diff to runs/fixloop/diff.md.
#
# usage: scripts/fixloop.sh BEFORE_REF AFTER_REF DATA_ROOT
#
# Environment:
#   PYTHON    interpreter with the dependencies (default: .venv/bin/python of the main checkout)
#   OUT       output folder (default: <repo>/runs/fixloop); before/ and after/ are replaced on each run
#   CACHE     model-output cache mode passed to the benchmark: live (default), replay or off
#   GPU_LOCK  lock file; when set, each benchmark run holds it via lockf -k (shared machines)
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 BEFORE_REF AFTER_REF DATA_ROOT" >&2
  exit 2
fi

repo=$(git rev-parse --show-toplevel)
main_root=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")
before_sha=$(git -C "$repo" rev-parse --verify "$1^{commit}")
after_sha=$(git -C "$repo" rev-parse --verify "$2^{commit}")
data_root=$(cd "$3" && pwd)
py=${PYTHON:-$main_root/.venv/bin/python}
[ -x "$py" ] || py=python3
out=${OUT:-$repo/runs/fixloop}
mkdir -p "$out"
out=$(cd "$out" && pwd)
cache=${CACHE:-live}

tmp=$(mktemp -d /tmp/scan2scope-fixloop.XXXXXX)
cleanup() {
  for side in before after; do
    if [ -d "$tmp/$side" ]; then
      git -C "$repo" worktree remove --force "$tmp/$side" >/dev/null 2>&1 || true
    fi
  done
  rm -rf "$tmp"
}
trap cleanup EXIT

git -C "$repo" worktree add --detach --quiet "$tmp/before" "$before_sha"
git -C "$repo" worktree add --detach --quiet "$tmp/after" "$after_sha"

# bench OUT_SIDE CODE_SIDE [extra args]: run the benchmark CLI from CODE_SIDE's worktree into $out/OUT_SIDE
bench() {
  local side=$1 code=$2
  shift 2
  local cmd=(env PYTHONPATH="$tmp/$code/src" "$py" -m scan2scope.cli bench "$data_root" --out "$out/$side"
             --cache "$cache" "$@")
  if [ -n "${GPU_LOCK:-}" ]; then
    cmd=(lockf -k "$GPU_LOCK" "${cmd[@]}")
  fi
  (cd "$tmp/$code" && "${cmd[@]}")
}

for side in before after; do
  rm -rf "${out:?}/$side"
  sha=$before_sha
  [ "$side" = after ] && sha=$after_sha
  echo "== $side: $(git -C "$repo" log -1 --format='%h %s' "$sha")"
  bench "$side" "$side"
done

# Score both runs with the after ref's harness and gates when those differ, so only the pipeline differs.
rescored=()
if ! git -C "$repo" diff --quiet "$before_sha" "$after_sha" -- src/scan2scope/bench bench/gates.yaml; then
  echo "== harness or gates changed between the refs: rescoring the before run with the after harness"
  bench before after --skip-run
  rescored=(--rescored)
fi

"$py" "$repo/scripts/compare_runs.py" "$out/before" "$out/after" --before-ref "$before_sha" \
  --after-ref "$after_sha" --repo "$repo" --out "$out/diff.md" "${rescored[@]+"${rescored[@]}"}"
