#!/usr/bin/env bash
# One fresh experiment, including its own proxy preparation, under one outer timer.
set -euo pipefail
unset PYTHONOPTIMIZE

case "${1:-}" in
  3|4) experiment="$1"; name="experiment$1" ;;
  5) experiment=5; name=experiment5_conservative ;;
  *) printf 'Usage: bash run_semijoin_experiment.sh {3|4|5} STUDY_DIR\n' >&2; exit 2 ;;
esac
study_dir="$(cd "${2:?STUDY_DIR is required}" && pwd -P)"
script="$(cd "$(dirname "$0")" && pwd -P)/$(basename "$0")"
source "$study_dir/settings.sh"
test "$(cd "$R" && pwd -P)" = "$study_dir"
cd "$BENCHMARK_DIR"
test "$(git rev-parse HEAD)" = "$(cat "$R/code_commit.txt")"
test "$N_ARTICLES" = 250
test "$REACTION_SET" = liver
test "$PROMPT" = paper
workflow="$R/${name}_workflow"

if [ "${3:-}" != --timed ]; then
  command -v setsid > /dev/null
  fresh "$workflow" "$R/$name"
  case "$experiment" in
    3) prefixes=(test_e5_score); scores=(test_e5) ;;
    4) prefixes=(test_biolord_score); scores=(test_biolord) ;;
    5) prefixes=(validation_biolord calibration test_biolord_exp5_score)
       scores=(validation_biolord thresholds test_biolord_exp5) ;;
  esac
  for prefix in "${prefixes[@]}" "$name"; do
    fresh "$R/${prefix}_wall_s.txt" "$R/${prefix}.log"
  done
  for score in "${scores[@]}"; do fresh "$R/$score.json"; done
  # mkdir also prevents two launchers from claiming the same experiment concurrently.
  mkdir "$workflow"
  completed=false
  timed_pid=''
  starting_timer=false
  pending_exit=''
  cancel() {
    if [ "$starting_timer" = true ]; then pending_exit=$1; return; fi
    trap '' HUP INT TERM
    if [ -n "$timed_pid" ]; then
      # Background jobs may inherit ignored INT. TERM stops the whole timed session.
      kill -TERM -- "-$timed_pid" 2>/dev/null || kill -TERM "$timed_pid" 2>/dev/null || true
      wait "$timed_pid" 2>/dev/null || true
    fi
    exit "$1"
  }
  finish() {
    local code=$? status=failed
    if [ "$code" -eq 0 ] && [ "$completed" = true ]; then status=complete; fi
    printf '{"experiment":%s,"status":"%s","exit_code":%s}\n' \
      "$experiment" "$status" "$code" > "$workflow/status.json.tmp"
    mv "$workflow/status.json.tmp" "$workflow/status.json"
  }
  trap finish EXIT
  trap 'cancel 129' HUP
  trap 'cancel 130' INT
  trap 'cancel 143' TERM
  printf '{"experiment":%s,"status":"preparing"}\n' "$experiment" > "$workflow/status.json"
  python - <<'PY'
import json, os
from pathlib import Path
choice = json.loads((Path(os.environ['R']) / 'embedding-choice.json').read_text(encoding='utf-8'))
for key in ('EMBED_DEVICE', 'EMBED_GPU_UUID', 'EMBED_HOST', 'EMBED_ORIGINAL_VISIBILITY'):
    assert choice[key] == os.environ[key], key
PY
  preflight "$name"
  check_embedding_context
  printf '{"experiment":%s,"status":"running"}\n' "$experiment" > "$workflow/status.json"
  starting_timer=true
  setsid /usr/bin/time -f '%e' -o "$workflow/total_wall_s.txt" \
    env SEMIJOIN_TIMED_CHILD="$name" bash "$script" "$experiment" "$study_dir" --timed \
    > "$workflow/execution.log" 2>&1 &
  timed_pid=$!
  starting_timer=false
  if [ -n "$pending_exit" ]; then cancel "$pending_exit"; fi
  wait "$timed_pid"
  timed_pid=''
  completed=true
  exit 0
fi

test "${SEMIJOIN_TIMED_CHILD:-}" = "$name"
test -d "$workflow"
score() {
  local model="$1" split="$2" output="$3" timing="$4"
  local -a limit=()
  if [ "$split" = test ]; then limit=(--n-articles "$N_ARTICLES"); fi
  /usr/bin/time -f '%e' -o "$R/${timing}_wall_s.txt" \
    env CUDA_VISIBLE_DEVICES="$EMBED_GPU_UUID" python semijoin_pruning.py score \
    --model "$model" --split "$split" "${limit[@]}" --device "$EMBED_DEVICE" \
    --output "$R/$output.json" > "$R/${timing}.log" 2>&1
  python - "$R/$output.json" "$model" "$split" <<'PY'
import os, sys
from pathlib import Path
from semijoin_baseline import LIVER_REACTIONS
from semijoin_pruning import MODELS, load_scores
scores = load_scores(Path(sys.argv[1]))
assert scores['scorer']['model'] == MODELS[sys.argv[2]]
assert scores['split'] == sys.argv[3]
assert scores['embedding_device'] == os.environ['EMBED_DEVICE']
assert scores['reactions'] == LIVER_REACTIONS
if sys.argv[3] == 'test':
    assert len(scores['articles']) == int(os.environ['N_ARTICLES'])
    assert [r['_qid'] for r in scores['articles'] if r['gold_matches']] == [29, 119, 134, 143, 148, 153, 158, 167, 198, 223]
print('Score input/panel/device checks passed:', sys.argv[1])
PY
}

case "$experiment" in
  3) score e5 test test_e5 test_e5_score; test_scores=test_e5 ;;
  4) score biolord test test_biolord test_biolord_score; test_scores=test_biolord ;;
  5)
    score biolord validation validation_biolord validation_biolord
    /usr/bin/time -f '%e' -o "$R/calibration_wall_s.txt" \
      python semijoin_pruning.py calibrate "$R/validation_biolord.json" \
      --output "$R/thresholds.json" > "$R/calibration.log" 2>&1
    score biolord test test_biolord_exp5 test_biolord_exp5_score
    test_scores=test_biolord_exp5
    ;;
esac
calibration=()
if [ "$experiment" = 5 ]; then calibration=(--calibration "$R/thresholds.json" --policy conservative); fi
/usr/bin/time -f '%e' -o "$R/${name}_wall_s.txt" \
  python semijoin_pruning.py run "$R/$test_scores.json" --experiment "$experiment" \
  "${calibration[@]}" --output-dir "$R/$name" > "$R/${name}.log" 2>&1
test -s "$R/$name/summary.json"
test -s "$R/$name/articles.json"
