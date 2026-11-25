# BioDEX 실험 3·4·5 서버 실행 절차

Linux Bash에서 아래 순서로 실행합니다. **70B GPU 서비스에서 3번, 4번, 5번을 각각 한 번 실측합니다.** E5와 BioLORD 임베딩은 CPU에서 계산합니다. 5번은 4번의 판정으로 두 정책을 먼저 재생한 뒤 보수적 정책만 실측합니다.

과거 실험 1·2번은 보고된 호출 수·논문 단위 F1·외부 시간을 별도로 인용합니다. 원본 요약이나 선택 CSV 없이 이 절차를 진행할 수 있습니다. 과거와 현재의 GPU·서버·캐시 조건을 확인하기 어려우면, 같은 환경에서 실행한 3~5번을 주 시간 비교로 사용합니다. [과거 결과와 출처](HISTORICAL_70B_RESULTS.ko.md) · [실험 설계와 지표](NEGATIVE_PRUNING_GUIDE.ko.md)

[서버 준비](#서버-준비) → [3번 E5](#실험-3-e5-정렬) → [4번 BioLORD](#실험-4-biolord-정렬) → [5번 pruning](#실험-5-pruning) → [품질 집계](#품질-집계) → [시간·순위 확인](#시간과-순위-확인) → [중단 시 재개](#중단-시-재개) · [백업](#결과-백업)

## 서버 준비

먼저 팀에서 사용할 수 있는 70B GPU 서비스와 실행 환경을 확인합니다. 과거 CloudLab 노드의 현재 접속 가능 여부는 미확인입니다. 서비스가 중지돼 있으면 해당 배포의 원래 기동 절차를 사용합니다.

| 항목 | 실행 조건 |
|---|---|
| GPU·모델 | 과거 기준은 V100 16GB 4장, `bartowski/Meta-Llama-3.1-70B-Instruct-GGUF`의 `Q4_K_M`입니다. 현재 서비스의 가중치·GPU·시작 설정을 대조해 차이를 기록합니다. |
| LLM 판정 | `PROMPT=paper`, 온도 0, 출력 최대 256토큰, `LLM_BATCH=8`, `EARLY_WORKERS=8`입니다. 시작 로그에서 슬롯당 context가 본문과 프롬프트·출력을 수용하는지 확인합니다. |
| 데이터 | BioDEX test 앞 250편, 본문 앞 8,000자, 간 라벨 10개를 사용합니다. Validation 전체는 5번의 임계값 보정에 사용합니다. |
| 점수 계산 | E5 test와 BioLORD test·validation 모두 `EMBED_DEVICE=cpu`로 고정합니다. 70B 서비스의 GPU 설정은 유지합니다. |

### Python 환경 활성화

서버에서 팀의 LOTUS Conda/venv를 활성화한 뒤 저장소 루트에서 시작합니다. 이후 모든 명령은 같은 Bash 세션에서 순서대로 실행합니다.

```bash
set -euo pipefail
git fetch origin
git switch feat/biolord-negative-pruning
git pull --ff-only
python -m pip install --no-deps -e .
python - <<'PY'
import sys, torch, lotus, datasets
print('python:', sys.version, 'lotus:', lotus.__file__)
print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
assert (3, 10) <= sys.version_info[:2] < (3, 13)
PY
cd benchmarks/biodex
```

<details>
<summary>새 Python 환경이 필요한 경우</summary>

저장소 루트에서 Python 3.10~3.12 환경에 `python -m pip install uv`, `uv export --locked --no-emit-project --no-hashes --all-extras -o /tmp/lotus-constraints.txt`, `uv pip install --python "$(command -v python)" -e . datasets --constraint /tmp/lotus-constraints.txt`를 차례로 실행합니다.

</details>

### 모델 ID와 출력 위치 기록

배포 설정에서 70B 가중치를 확인하고 **`LLM_MODEL`에 `hosted_vllm/<실제 모델 ID>`를 지정해 export합니다.** 기본 연결 주소는 `http://127.0.0.1:8080/v1`입니다. 다른 호스트의 서비스라면 `LLM_API_BASE`도 지정합니다. 인증이 필요하면 기존 `HOSTED_VLLM_API_KEY` 환경값을 사용합니다. 인증 정보는 endpoint나 보존 파일에 넣지 않습니다.

각 live run에 적용할 서버 캐시 절차를 먼저 정합니다. 팀의 표준 초기화 절차를 쓴다면 `SERVER_CACHE_POLICY`에 그 내용을 기록합니다. 실행 중인 서비스를 그대로 쓰면 아래 기본값이 초기 KV 캐시 상태를 통제하지 않았음을 기록합니다. 아래 블록은 **이번 비교에 사용할 새 폴더 `R`을 한 번만** 만듭니다.

<details>
<summary>서버 파일과 설정을 추가로 기록할 때</summary>

클라이언트에서 읽을 수 있는 실제 GGUF와 llama-server 경로가 있으면 블록 실행 전에 `MODEL_GGUF`, `LLAMA_SERVER`로 지정합니다. 두 파일이 있으면 SHA를 저장합니다. 확인한 가중치 revision·양자화·서버 버전·시작 명령도 남기려면 아래 `server-config.txt` 생성 블록에 추가합니다. `LLM_SERVER_ID`는 이렇게 기록한 내용의 식별자이며, alias나 식별자만으로 과거와 같은 가중치인지는 확인할 수 없습니다.

</details>

```bash
export N_ARTICLES=250
export REACTION_SET=liver
export PROMPT=paper
: "${LLM_MODEL:?배포에서 확인한 70B 모델을 hosted_vllm/모델ID 형식으로 지정하세요}"
export LLM_MODEL
export LLM_API_BASE="${LLM_API_BASE:-http://127.0.0.1:8080/v1}"
case "$LLM_MODEL" in
  hosted_vllm/llama3.1-8b) printf '70B 서비스의 모델 ID를 지정하세요.\n' >&2; exit 1 ;;
  hosted_vllm/?*) ;;
  *) printf '로컬 경로는 hosted_vllm/<기존 서버 모델 ID>를 사용하세요. 가중치는 변경하지 않습니다.\n' >&2; exit 1 ;;
esac
export LLM_BATCH=8
export EARLY_WORKERS=8
export EMBED_DEVICE=cpu
export SERVER_CACHE_POLICY="${SERVER_CACHE_POLICY:-existing_server_initial_kv_uncontrolled}"
export BENCHMARK_DIR="$(pwd)"
export R="$(pwd)/pruning_results/${REACTION_SET}_${PROMPT}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$(dirname "$R")"
mkdir "$R"
curl_auth=()
if [ -n "${HOSTED_VLLM_API_KEY:-}" ]; then
  curl_auth=(-H "Authorization: Bearer $HOSTED_VLLM_API_KEY")
fi
curl "${curl_auth[@]}" --fail --silent --show-error --max-time 30 \
  "${LLM_API_BASE%/}/models" > "$R/existing_server_models.json"
if [ -f "${MODEL_GGUF:-}" ] && [ -r "${MODEL_GGUF:-}" ] && \
   [ -f "${LLAMA_SERVER:-}" ] && [ -r "${LLAMA_SERVER:-}" ]; then
  sha256sum "$MODEL_GGUF" "$LLAMA_SERVER" > "$R/server-files.sha256"
fi
{
  printf 'source=current_70b_gpu_service\n'
  printf 'model_alias=%s\nendpoint=%s\ncache_policy=%s\n' \
    "$LLM_MODEL" "$LLM_API_BASE" "$SERVER_CACHE_POLICY"
  if [ -f "$R/server-files.sha256" ]; then
    cat "$R/server-files.sha256"
  else
    printf 'weights/build/launch_details=not_recorded; keep_existing_server_settings\n'
  fi
} > "$R/server-config.txt"
export LLM_SERVER_ID="$(sha256sum "$R/server-config.txt" | cut -d' ' -f1)"
printf 'set -euo pipefail\n' > "$R/settings.sh"
for name in R REACTION_SET PROMPT LLM_MODEL LLM_API_BASE LLM_BATCH EARLY_WORKERS \
            EMBED_DEVICE N_ARTICLES SERVER_CACHE_POLICY LLM_SERVER_ID BENCHMARK_DIR; do
  printf 'export %s=%q\n' "$name" "${!name}"
done >> "$R/settings.sh"
git rev-parse HEAD > "$R/code_commit.txt"
python -m pip freeze > "$R/python_packages.txt"
printf 'Study directory: %s\n' "$R"
fresh() {
  for path in "$@"; do
    if [ -e "$path" ]; then
      printf 'Existing output: %s. Reuse it or choose a new study directory.\n' "$path" >&2
      return 1
    fi
  done
}
```

출력된 `Study directory`를 보관합니다. GPU 서버에서 `nvidia-smi -L` 출력을 `$R/gpu.txt`에 저장합니다. 클라이언트가 다른 호스트에 있다면 GPU 서버 출력을 가져와 기록합니다.

### 연결 점검 함수 등록

아래 블록을 한 번 실행합니다. 이후 각 `preflight`가 실제 모델 ID와 LOTUS를 통한 짧은 요청의 응답을 확인합니다. 이 요청은 실험 호출 수와 시간에서 제외합니다. 세 실험에 같은 캐시 절차와 사전 점검을 적용합니다.

```bash
preflight() {
  local run_name="$1"
  case "$LLM_MODEL" in
    hosted_vllm/?*) ;;
    *) printf '로컬 경로는 hosted_vllm/<기존 서버 모델 ID>가 필요합니다.\n' >&2; return 1 ;;
  esac
  fresh "$R/${run_name}_preflight.log" "$R/${run_name}_models.json"
  if [ -f "$R/server-files.sha256" ]; then
    sha256sum --check --status "$R/server-files.sha256"
  fi
  test "$(sha256sum "$R/server-config.txt" | cut -d' ' -f1)" = "$LLM_SERVER_ID"
  local -a auth=()
  if [ -n "${HOSTED_VLLM_API_KEY:-}" ]; then
    auth=(-H "Authorization: Bearer $HOSTED_VLLM_API_KEY")
  fi
  curl "${auth[@]}" --fail --silent --show-error --max-time 30 \
    "${LLM_API_BASE%/}/models" > "$R/${run_name}_models.json"
  python - "$R/${run_name}_models.json" <<'PY'
import json, os, sys
ids = [entry['id'] for entry in json.load(open(sys.argv[1]))['data']]
expected = os.environ['LLM_MODEL'].split('/', 1)[1]
assert expected in ids, (expected, ids)
print('Server model alias:', expected)
PY
  python - <<'PY' > "$R/${run_name}_preflight.log" 2>&1
import semijoin_baseline as baseline
lm = baseline.configure(with_retriever=False)
out = lm([[{'role': 'user', 'content': 'Reply with exactly True.'}]],
         show_progress_bar=False, timeout=60, num_retries=0)
text = out.outputs[0].strip()
assert text and 'true' in text.lower(), repr(text)
print('LOTUS local LLM route OK:', baseline.LLM_MODEL, repr(text))
PY
}
declare -f fresh preflight >> "$R/settings.sh"
```

### 시간 측정 전 데이터와 모델 파일 준비

본 측정과 같은 사용자·캐시 경로에서 실행합니다. Test·validation 데이터와 두 모델의 가중치·tokenizer를 미리 준비합니다. 종료 후 `$R/preparation.log`에 split별 건수와 데이터 식별값, 모델별 최대 입력 길이가 남습니다.

```bash
fresh "$R/preparation.log"
python - <<'PY' > "$R/preparation.log" 2>&1
from datasets import load_dataset
from transformers import AutoTokenizer
from lotus.models import SentenceTransformersRM
from semijoin_pruning import MODELS

for split in ('test', 'validation'):
    data = load_dataset('BioDEX/BioDEX-Reactions', split=split)
    print(split, len(data), data._fingerprint)
for model in MODELS.values():
    AutoTokenizer.from_pretrained(model)
    rm = SentenceTransformersRM(model=model, device='cpu', normalize_embeddings=True)
    print(model, rm.transformer.max_seq_length)
    del rm
PY
```

## 실험 3 E5 정렬

먼저 CPU에서 E5 test 점수를 만들고 이 명령의 경과시간을 저장합니다.

```bash
fresh "$R/test_e5.json" "$R/test_e5_score_wall_s.txt" "$R/test_e5_score.log"
/usr/bin/time -f '%e' -o "$R/test_e5_score_wall_s.txt" \
  python semijoin_pruning.py score --model e5 --split test --n-articles "$N_ARTICLES" \
  --device "$EMBED_DEVICE" --output "$R/test_e5.json" > "$R/test_e5_score.log" 2>&1
```

LLM을 호출하기 전에 0부터 시작하는 test 행 번호, 양성 10편의 인덱스, 간 라벨 순서를 과거 기록과 대조합니다. 대조가 실패하면 먼저 입력을 확인합니다. 일치하더라도 과거 250편의 원문 전체와 데이터 revision까지 식별되는 것은 아닙니다.

```bash
python - "$R/test_e5.json" <<'PY'
import json, sys
from pathlib import Path
scores = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
rows = scores['articles']
assert scores['split'] == 'test' and [row['_qid'] for row in rows] == list(range(250))
assert [row['_qid'] for row in rows if row['gold_matches']] == [29, 119, 134, 143, 148, 153, 158, 167, 198, 223]
assert scores['reactions'] == [
    'Drug-induced liver injury', 'Hepatic function abnormal', 'Hepatotoxicity', 'Liver injury',
    'Hepatic failure', 'Hepatic enzyme increased', 'Acute hepatic failure', 'Hepatitis',
    'Immune-mediated hepatitis', 'Hepatic necrosis',
]
print('250 test rows, 10 historical gold-positive indices and ordered liver panel match')
PY
```

연결 점검 후 E5 순서로 조기 종료를 수행합니다. 3번에는 임계값을 지정하지 않습니다. `summary.json`과 `articles.json`이 생성되면 4번으로 넘어갑니다.

```bash
preflight experiment3
fresh "$R/experiment3" "$R/experiment3_wall_s.txt" "$R/experiment3.log"
/usr/bin/time -f '%e' -o "$R/experiment3_wall_s.txt" \
  python semijoin_pruning.py run "$R/test_e5.json" --experiment 3 \
  --output-dir "$R/experiment3" > "$R/experiment3.log" 2>&1
test -s "$R/experiment3/summary.json"
test -s "$R/experiment3/articles.json"
```

## 실험 4 BioLORD 정렬

CPU에서 BioLORD test·validation 점수를 만듭니다. **4번의 test LLM 판정을 보기 전에** validation의 정답과 점수로 5번의 두 임계값을 고정합니다. 아래 검사는 두 모델의 test 입력·조각·라벨 일치와 세 점수 파일의 CPU 사용 여부를 확인합니다. 불일치 시 기존 점수를 보존하고 입력이나 모델 버전을 확인합니다.

```bash
test -s "$R/experiment3/summary.json"
fresh "$R/test_biolord.json" "$R/test_biolord_score_wall_s.txt" "$R/test_biolord_score.log"
/usr/bin/time -f '%e' -o "$R/test_biolord_score_wall_s.txt" \
  python semijoin_pruning.py score --model biolord --split test --n-articles "$N_ARTICLES" \
  --device "$EMBED_DEVICE" --output "$R/test_biolord.json" > "$R/test_biolord_score.log" 2>&1
fresh "$R/validation_biolord.json" "$R/validation_biolord_wall_s.txt" "$R/validation_biolord.log"
/usr/bin/time -f '%e' -o "$R/validation_biolord_wall_s.txt" \
  python semijoin_pruning.py score --model biolord --split validation --device "$EMBED_DEVICE" \
  --output "$R/validation_biolord.json" > "$R/validation_biolord.log" 2>&1
fresh "$R/thresholds.json"
python semijoin_pruning.py calibrate "$R/validation_biolord.json" --output "$R/thresholds.json"
python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['R'])
read = lambda name: json.loads((root / name).read_text(encoding='utf-8'))
e5, bio, validation, calibration = map(read, ['test_e5.json', 'test_biolord.json', 'validation_biolord.json', 'thresholds.json'])
for key in ['input_sha256', 'chunks_sha256', 'reactions']:
    assert e5[key] == bio[key], key
assert e5['scorer']['chunking'] == bio['scorer']['chunking']
assert all(scores['embedding_device'] == 'cpu' for scores in (e5, bio, validation))
assert bio['scorer'] == validation['scorer'] == calibration['scorer']
assert bio['reactions'] == calibration['reactions']
assert len(e5['articles']) == len(bio['articles']) == int(os.environ['N_ARTICLES'])
print('Shared abstracts/chunks/panel and calibration OK')
print(json.dumps(calibration['policies'], indent=2))
PY
```

같은 LLM 설정에서 BioLORD 순서로 조기 종료를 수행합니다. 4번에도 임계값을 지정하지 않습니다. 완료된 결과와 BioLORD 점수 파일을 그대로 5번에서 사용합니다.

```bash
preflight experiment4
fresh "$R/experiment4" "$R/experiment4_wall_s.txt" "$R/experiment4.log"
/usr/bin/time -f '%e' -o "$R/experiment4_wall_s.txt" \
  python semijoin_pruning.py run "$R/test_biolord.json" --experiment 4 \
  --output-dir "$R/experiment4" > "$R/experiment4.log" 2>&1
test -s "$R/experiment4/summary.json"
test -s "$R/experiment4/articles.json"
```

## 실험 5 pruning

먼저 4번의 완료 기록을 바탕으로 `conservative`와 `retain_95`를 각각 재생합니다. 두 파일은 정책별 호출 수와 품질을 비교하는 데 쓰며 추가 LLM 호출은 없습니다.

```bash
test -s "$R/experiment4/summary.json"
fresh "$R/experiment5_conservative_replay.json" "$R/experiment5_retain95_replay.json"
python semijoin_pruning.py replay "$R/test_biolord.json" \
  --calibration "$R/thresholds.json" --policy conservative \
  --source-dir "$R/experiment4" --output "$R/experiment5_conservative_replay.json"
python semijoin_pruning.py replay "$R/test_biolord.json" \
  --calibration "$R/thresholds.json" --policy retain_95 \
  --source-dir "$R/experiment4" --output "$R/experiment5_retain95_replay.json"
```

실제 시간은 사전에 정한 보수적 정책 `conservative`로 한 번 측정합니다. 4번과 동일한 BioLORD 점수 파일, 고정한 validation 임계값을 사용합니다.

```bash
preflight experiment5_conservative
fresh "$R/experiment5_conservative" "$R/experiment5_conservative_wall_s.txt" "$R/experiment5_conservative.log"
/usr/bin/time -f '%e' -o "$R/experiment5_conservative_wall_s.txt" \
  python semijoin_pruning.py run "$R/test_biolord.json" --experiment 5 \
  --calibration "$R/thresholds.json" --policy conservative \
  --output-dir "$R/experiment5_conservative" > "$R/experiment5_conservative.log" 2>&1
test -s "$R/experiment5_conservative/summary.json"
test -s "$R/experiment5_conservative/articles.json"
```

모두 생략된 경우에도 `articles.json`과 `summary.json`이 만들어집니다. 이 경우 `progress.jsonl`이 없어도 정상입니다. `retain_95`의 실제 실행시간은 미측정으로 표시합니다.

## 품질 집계

아래 명령에 실측 세 건과 4번을 원본으로 한 replay 두 건을 전달합니다. 각 실측 폴더에는 해당 실행에서 사용한 점수 파일을 짝지어 줍니다. 집계에는 LLM 호출이 없습니다.

```bash
fresh "$R/comparison.json"
python semijoin_pruning.py report --scores "$R/test_biolord.json" \
  --run "$R/test_e5.json" "$R/experiment3" \
  --run "$R/test_biolord.json" "$R/experiment4" \
  --run "$R/test_biolord.json" "$R/experiment5_conservative" \
  --replay "$R/experiment5_conservative_replay.json" \
  --replay "$R/experiment5_retain95_replay.json" \
  --output "$R/comparison.json"
```

`comparison.json`의 `results[].vs_labels`에서 논문 단위 TP·FP·FN과 정답 대비 precision·recall·F1을 읽습니다. `mode`와 `lm_calls_kind`로 실측 호출 수와 replay가 추정한 정책 비용을 구분합니다. **예상 결과는 `missing_experiments: [1, 2]`와 모든 행의 `vs_join_full: null`입니다.** 과거의 전체 선택 ID가 없어 1번과의 논문별 일치도를 계산할 수 없습니다.

## 시간과 순위 확인

같은 `R`에서 세 실측의 설정과 점수 식별자를 대조하고 각 방법의 호출 수, F1, recall, 정답 양성 논문의 첫 gold witness 평균 순위를 읽습니다. 재개한 실험은 전체 시간이 미측정으로 출력됩니다.

```bash
python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['R'])
read = lambda path: json.loads(path.read_text(encoding='utf-8'))
names = ['experiment3', 'experiment4', 'experiment5_conservative']
ms = [read(root / name / 'manifest.json') for name in names]
assert all(m['model'] == os.environ['LLM_MODEL'] and m['workers'] == 8 and m['llm_batch'] == 8 for m in ms)
assert all(m['max_tokens'] == 256 and m['temperature'] == 0 for m in ms)
for key in ['model', 'instruction', 'max_tokens', 'temperature', 'workers', 'llm_batch', 'endpoint_sha256', 'llm_server_id']:
    assert len({json.dumps(m[key], sort_keys=True) for m in ms}) == 1, key
assert ms[1]['scores_sha256'] == ms[2]['scores_sha256']
for name, score_time in zip(names, ['test_e5_score_wall_s.txt', 'test_biolord_score_wall_s.txt', 'test_biolord_score_wall_s.txt']):
    summary = read(root / name / 'summary.json')
    rows = read(root / name / 'articles.json')
    assert [r['_qid'] for r in rows if r['gold_positive']] == [29, 119, 134, 143, 148, 153, 158, 167, 198, 223]
    ranks = [r['first_gold_witness_rank'] for r in rows if r['gold_positive']]
    print(name, 'calls', summary['lm_calls'], 'F1', summary['f1'], 'recall', summary['recall'])
    print('mean first gold witness rank', sum(ranks) / len(ranks) if ranks else None)
    if summary['resumed']:
        print('test score + run wall: unavailable (resumed run)')
        continue
    score_wall = float((root / score_time).read_text().strip())
    run_wall = float((root / f'{name}_wall_s.txt').read_text().strip())
    assert run_wall + 0.1 >= summary['invocation_time_s'], 'Wall timer may have been overwritten by a completed rerun'
    print('test score + run wall seconds', score_wall + run_wall)
PY
```

| 방법 | Test 처리 시간 |
|---|---|
| 3번 | E5 test 점수 생성 시간 + 3번 실측 시간 |
| 4번 | BioLORD test 점수 생성 시간 + 4번 실측 시간 |
| 5번 실측 | 같은 BioLORD test 점수 생성 시간 + 5번 실측 시간 |

위 합계는 두 명령을 각각 외부 타이머로 측정한 값입니다. BioLORD test 점수는 한 번만 만들고, 각 방법을 독립적으로 실행할 때의 비용으로 4번과 5번에 1회분씩 반영합니다. Validation 보정, 서버 기동, 사전 준비·연결 점검은 따로 기록합니다. 점수 계산 시 모델을 메모리에 올리는 시간과 온라인 메타데이터 조회는 측정에 포함되며, 파일을 실제로 추가 다운로드했다면 ‘네트워크·다운로드 포함’으로 표시합니다.

과거 전체 조인 7,282초와 고정 순서 조기 종료 1,896초는 실행 중인 GPU 서버에 연결한 Python 명령 시작부터 종료까지의 경과시간입니다. GPU 종류·수, 가중치·서버·동시성·캐시 조건이 확인될 때만 과거 대비 속도배율을 해석합니다. [과거 시간의 출처와 범위](HISTORICAL_70B_RESULTS.ko.md)

## 중단 시 재개

**중단된 경우에만** 이 단계를 사용합니다. 새 터미널에서 원래 Python 환경과 인증을 활성화하고 `source /실제/기존/R/settings.sh` 후 `cd "$BENCHMARK_DIR"`를 실행합니다. 원래 점수·출력 폴더·실험 번호를 사용하면 완료된 논문은 건너뜁니다. 아래는 첫 번째 3번 재개의 예입니다.

```bash
preflight experiment3_resume1
fresh "$R/experiment3_resume1_wall_s.txt" "$R/experiment3_resume1.log"
/usr/bin/time -f '%e' -o "$R/experiment3_resume1_wall_s.txt" \
  python semijoin_pruning.py run "$R/test_e5.json" --experiment 3 \
  --output-dir "$R/experiment3" > "$R/experiment3_resume1.log" 2>&1
```

4번은 원래 BioLORD 점수와 실험 번호를, 5번은 원래 calibration·policy까지 사용합니다. 재개할 때마다 사전 점검·로그·타이머에 새 이름을 붙입니다. 미완료 논문의 호출은 일부 빠질 수 있어 재개 시간을 더해 무중단 실측 시간으로 쓰지 않습니다. 전체 시간이 필요하면 보존한 점수를 사용해 새 출력 폴더에서 다시 실측합니다.

## 결과 백업

노드 예약이 끝나기 전에 **이번 `R` 디렉터리 전체**를 팀의 지속 저장소에 보관합니다. `comparison.json`의 품질 수치를 검토할 때에는 점수·임계값, 실험별 manifest·progress·articles·summary, replay 결과가 함께 필요합니다. 시간을 검토할 때에는 각 외부 타이머·실행 로그, 준비 로그, 서버 구성과 장비·패키지 기록을 함께 사용합니다.
