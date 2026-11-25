# BioDEX 실험 3·4·5 서버 실행 절차

Linux Bash에서 아래 순서로 실행합니다. **70B GPU 서비스에서 3번, 4번, 5번을 각각 한 번 실측합니다.** 임베딩은 사전 점검을 통과한 GPU 한 장을 우선 사용하고, 사용할 GPU가 없거나 시험에 실패하면 CPU를 사용합니다. **3번 시작 전에 한 번 결정해 E5 test·BioLORD test·validation에 같은 장치를 유지합니다.** 5번은 4번의 판정으로 두 정책을 먼저 재생한 뒤 보수적 정책만 실측합니다.

과거 실험 1·2번은 보고된 호출 수·논문 단위 F1·외부 시간을 별도로 인용합니다. 원본 요약이나 선택 CSV 없이 이 절차를 진행할 수 있습니다. 과거와 현재의 GPU·서버·캐시 조건을 확인하기 어려우면, 같은 환경에서 실행한 3~5번을 주 시간 비교로 사용합니다. [과거 결과와 출처](HISTORICAL_70B_RESULTS.ko.md) · [실험 설계와 지표](NEGATIVE_PRUNING_GUIDE.ko.md)

[서버 준비](#서버-준비) → [3번 E5](#실험-3-e5-정렬) → [4번 BioLORD](#실험-4-biolord-정렬) → [5번 pruning](#실험-5-pruning) → [품질 집계](#품질-집계) → [시간·순위 확인](#시간과-순위-확인) → [중단 시 재개](#중단-시-재개) · [백업](#결과-백업)

## 서버 준비

먼저 팀에서 사용할 수 있는 70B GPU 서비스와 실행 환경을 확인합니다. 과거 CloudLab 노드의 현재 접속 가능 여부는 미확인입니다. 서비스가 중지돼 있으면 해당 배포의 원래 기동 절차를 사용합니다.

| 항목 | 실행 조건 |
|---|---|
| GPU·모델 | 과거 기준은 V100 16GB 4장, `bartowski/Meta-Llama-3.1-70B-Instruct-GGUF`의 `Q4_K_M`입니다. 현재 서비스의 가중치·GPU·시작 설정을 대조해 차이를 기록합니다. |
| LLM 판정 | `PROMPT=paper`, 온도 0, 출력 최대 256토큰, `LLM_BATCH=8`, `EARLY_WORKERS=8`입니다. 시작 로그에서 슬롯당 context가 본문과 프롬프트·출력을 수용하는지 확인합니다. |
| 데이터 | BioDEX test 앞 250편, 본문 앞 8,000자, 간 라벨 10개를 사용합니다. Validation 전체는 5번의 임계값 보정에 사용합니다. |
| 점수 계산 | 아래 사전 시험 후 `EMBED_DEVICE=cpu` 또는 `cuda:0`을 고정합니다. GPU이면 세 점수 명령에 같은 물리 GPU UUID를 지정합니다. 70B 서비스의 GPU 배정·설정은 유지합니다. |

팀원이 알려준 V100 8장은 서버에 설치된 수량이며, 과거 기록의 70B 사용 GPU 4장과 구분합니다. 실제 배정·사용 장수와 장당 VRAM을 확인합니다. RAM 48/96GB라는 설명만으로 GPU 여유를 판단하지 않습니다. 임베딩을 위해 70B를 다른 GPU에 재배치하지 않습니다.

### Python 환경 활성화

서버에서 팀의 LOTUS Conda/venv를 활성화한 뒤 저장소 루트에서 시작합니다. 이후 모든 명령은 같은 Bash 세션에서 순서대로 실행합니다.

```bash
set -euo pipefail
unset PYTHONOPTIMIZE
git fetch origin
git switch feat/biolord-negative-pruning
git pull --ff-only
python -m pip install --no-deps -e .
python - <<'PY'
import sys, torch, lotus, datasets
if not __debug__:
    raise SystemExit('Python 최적화 모드를 해제해야 입력·장치 검사가 실행됩니다.')
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
printf 'set -euo pipefail\nunset PYTHONOPTIMIZE\n' > "$R/settings.sh"
for name in R REACTION_SET PROMPT LLM_MODEL LLM_API_BASE LLM_BATCH EARLY_WORKERS \
            N_ARTICLES SERVER_CACHE_POLICY LLM_SERVER_ID BENCHMARK_DIR; do
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

출력된 `Study directory`를 보관합니다. 70B가 실행되는 GPU 서버에서 `nvidia-smi`와 `nvidia-smi -L` 출력을 각각 `$R/llm_gpu_processes.txt`, `$R/gpu.txt`에 저장합니다. 클라이언트가 다른 호스트라면 해당 서버 출력을 가져옵니다. 아래 임베딩 점검은 **LLM 접속 주소가 아니라 실제 score 명령을 실행할 컴퓨터**에서 합니다.

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

### 임베딩 장치 점검·선택

**이 단계는 점수 생성·3번 LLM 평가 전에 한 번만 실행하며 사전 준비 비용입니다.** 먼저 같은 사용자·컨테이너·작업 할당에서 GPU별 UUID·전체/남은 메모리·프로세스를 확인합니다. `nvidia-smi`에 보인다는 이유만으로 다른 작업에 배정된 GPU를 사용하지 않습니다. 기존 `CUDA_VISIBLE_DEVICES`와 스케줄러의 GPU 할당을 확인하고, 사용 권한이 있으며 70B에 영향을 주지 않는 한 장만 후보로 선택합니다. 할당이나 여유가 불명확하면 CPU로 진행합니다.

```bash
fresh "$R/embedding_inventory.log"
{
  hostname
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES-<unset>}"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free --format=csv || true
    nvidia-smi || true
  fi
  python - <<'PY'
import torch
print('torch', torch.__version__, 'CUDA runtime', torch.version.cuda)
try:
    available = torch.cuda.is_available()
    print('CUDA available', available)
    if available:
        for index in range(torch.cuda.device_count()):
            print('visible device', index, torch.cuda.get_device_properties(index))
except Exception as error:
    print('CUDA inventory unavailable; CPU remains eligible:', repr(error))
PY
} > "$R/embedding_inventory.log" 2>&1
```

GPU를 시험할 경우 **현재 작업에 배정된 GPU의 전체 UUID**를 `EMBED_CANDIDATE_GPU_UUID`로 export합니다. 예: `export EMBED_CANDIDATE_GPU_UUID=GPU-실제전체UUID`. 숫자 인덱스나 예시 UUID를 그대로 넣지 않습니다. 사용할 GPU가 없으면 `unset EMBED_CANDIDATE_GPU_UUID`로 명시합니다. 후보를 정하기 전에 모델 파일 준비가 정상 종료했는지도 확인합니다.

아래 시험은 E5와 BioLORD를 **각각 별도 프로세스**로 로드하고 공통 길이에 가까운 초록 조각 64개와 liver 라벨을 실제로 임베딩합니다. 모델의 실제 장치, 출력 유효성, 소요시간과 GPU 시험 프로세스의 PyTorch 최대 메모리를 기록합니다. 최대 메모리는 70B를 포함한 GPU 전체 사용량이 아닙니다. 두 GPU 시험이 모두 성공하면 GPU, 실패하면 원인을 로그에 남긴 채 두 CPU 시험까지 통과해야 CPU를 선택합니다. 데이터·모델 파일 오류는 먼저 해결하며 GPU 메모리 부족으로 단정하지 않습니다. 시험 속도는 참고값이고 전체 처리시간 개선은 본 실측으로 확인합니다.

```bash
fresh "$R/embedding-choice.json" "$R/embedding_candidate_check.log" \
      "$R/embedding_probe_gpu_e5.log" "$R/embedding_probe_gpu_biolord.log" \
      "$R/embedding_probe_cpu_e5.log" "$R/embedding_probe_cpu_biolord.log"
gpu_in_current_allocation() {
  python - "$1" <<'PY'
import re, sys
import torch
candidate = sys.argv[1]
pattern = r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}'
try:
    assert re.fullmatch(pattern, candidate), 'A full GPU UUID is required.'
    assert torch.cuda.is_available(), 'CUDA is unavailable in the original job allocation.'
    visible = []
    for index in range(torch.cuda.device_count()):
        value = getattr(torch.cuda.get_device_properties(index), 'uuid', None)
        assert value is not None, 'This environment cannot identify the physical GPU UUID.'
        value = str(value)
        visible.append(value if value.startswith('GPU-') else 'GPU-' + value)
    print('original job visible UUIDs:', visible, 'candidate:', candidate)
    assert candidate in visible, 'The candidate is outside the original visible job allocation.'
except Exception as error:
    print('GPU candidate rejected:', repr(error))
    raise SystemExit(1)
PY
}
probe_embeddings() {
  local device="$1" visible="$2" tag="$3" model
  for model in e5 biolord; do
    fresh "$R/embedding_probe_${tag}_${model}.log" || return 1
    env CUDA_VISIBLE_DEVICES="$visible" EMBED_PROBE_DEVICE="$device" EMBED_PROBE_MODEL="$model" \
      python - <<'PY' > "$R/embedding_probe_${tag}_${model}.log" 2>&1 || return 1
import os, time
import numpy as np
import torch
from transformers import AutoTokenizer
from lotus.models import SentenceTransformersRM
from semijoin_baseline import LIVER_REACTIONS
from semijoin_pruning import MODELS, abstract_chunks

device, name = os.environ['EMBED_PROBE_DEVICE'], os.environ['EMBED_PROBE_MODEL']
print('requested', device, 'visibility', os.environ['CUDA_VISIBLE_DEVICES'], flush=True)
if device != 'cpu':
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    print('GPU', torch.cuda.get_device_properties(0), flush=True)
    torch.cuda.reset_peak_memory_stats()
started = time.perf_counter()
rm = SentenceTransformersRM(model=MODELS[name], device=device, normalize_embeddings=True)
assert str(rm.transformer.device) == device, rm.transformer.device
tokenizers = {key: rm.transformer.tokenizer if key == name else AutoTokenizer.from_pretrained(model)
              for key, model in MODELS.items()}
chunk = abstract_chunks('The patient developed drug-induced liver injury after treatment. ' * 80, tokenizers)[0]
prefix = 'passage: ' if name == 'e5' else ''
vectors = rm([prefix + chunk.strip()] * 64)
labels = rm([('query: ' if name == 'e5' else '') + label for label in LIVER_REACTIONS])
assert vectors.shape[0] == 64 and labels.shape[0] == len(LIVER_REACTIONS)
assert np.isfinite(vectors).all() and np.isfinite(labels).all()
assert (np.linalg.norm(vectors, axis=1) > 0).all() and (np.linalg.norm(labels, axis=1) > 0).all()
print('model', name, 'actual device', rm.transformer.device,
      'dtype', next(rm.transformer.parameters()).dtype,
      'probe_seconds', time.perf_counter() - started)
if device != 'cpu':
    print('probe_peak_allocated_bytes', torch.cuda.max_memory_allocated(),
          'probe_peak_reserved_bytes', torch.cuda.max_memory_reserved())
PY
  done
}
export EMBED_DEVICE=cpu
export EMBED_GPU_UUID=""
export EMBED_SELECTION_REASON=no_assigned_spare_gpu
if [ -n "${EMBED_CANDIDATE_GPU_UUID:-}" ]; then
  if gpu_in_current_allocation "$EMBED_CANDIDATE_GPU_UUID" > "$R/embedding_candidate_check.log" 2>&1; then
    if probe_embeddings cuda:0 "$EMBED_CANDIDATE_GPU_UUID" gpu; then
      export EMBED_DEVICE=cuda:0
      export EMBED_GPU_UUID="$EMBED_CANDIDATE_GPU_UUID"
      export EMBED_SELECTION_REASON=both_gpu_probes_passed
    else
      export EMBED_SELECTION_REASON=gpu_probe_failed_cpu_probes_required
      printf 'GPU 시험 실패: 로그를 확인합니다. CPU 시험 중 하나라도 실패하면 중단합니다.\n' >&2
    fi
  else
    export EMBED_SELECTION_REASON=gpu_candidate_not_verified_in_job_allocation
    printf 'GPU 후보를 현재 작업의 전체 UUID로 확인하지 못했습니다. CPU를 시험합니다.\n' >&2
  fi
fi
if [ "$EMBED_DEVICE" = cpu ]; then
  probe_embeddings cpu "" cpu
fi
export EMBED_HOST="$(hostname)"
export EMBED_ORIGINAL_VISIBILITY="${CUDA_VISIBLE_DEVICES-<unset>}"
check_embedding_context() {
  test "$(hostname)" = "$EMBED_HOST" || return 1
  if [ "$EMBED_DEVICE" = cuda:0 ]; then
    test "${CUDA_VISIBLE_DEVICES-<unset>}" = "$EMBED_ORIGINAL_VISIBILITY" || {
      printf 'GPU 작업 할당/가시성이 바뀌었습니다. 원래 UUID의 사용 권한을 다시 확인하세요.\n' >&2
      return 1
    }
    gpu_in_current_allocation "$EMBED_GPU_UUID" || return 1
  fi
}
python - <<'PY'
import json, os
from pathlib import Path
keys = ['EMBED_DEVICE', 'EMBED_GPU_UUID', 'EMBED_SELECTION_REASON', 'EMBED_HOST', 'EMBED_ORIGINAL_VISIBILITY']
choice = {key: os.environ[key] for key in keys}
(Path(os.environ['R']) / 'embedding-choice.json').write_text(json.dumps(choice, indent=2), encoding='utf-8')
print(json.dumps(choice, indent=2))
PY
for name in EMBED_DEVICE EMBED_GPU_UUID EMBED_SELECTION_REASON EMBED_HOST EMBED_ORIGINAL_VISIBILITY; do
  printf 'export %s=%q\n' "$name" "${!name}"
done >> "$R/settings.sh"
declare -f gpu_in_current_allocation check_embedding_context >> "$R/settings.sh"
```

`cuda:0`은 원래 작업의 가시 목록에서 전체 UUID를 확인한 뒤, 후보 한 장만 보이게 한 **해당 임베딩 자식 프로세스의 번호**입니다. UUID를 식별할 수 없는 환경이면 CPU로 진행합니다. 70B 서버나 현재 셸의 GPU 가시성을 바꾸지 않습니다. CPU이면 임베딩 자식 프로세스에만 GPU를 숨깁니다. 선택 이후에는 장치·UUID·정밀도를 바꾸거나 점수 파일을 CPU/GPU 사이에 섞지 않습니다. 장치 선택 단계가 중단되어 시험 로그가 남았다면 보존하고 새 `R`에서 준비합니다. 본 점수 생성 중 GPU 오류가 나면 그대로 중단하고 원인을 확인합니다. 3번 이후 CPU로 바꿔야 한다면 기존 결과를 보존하고 새 `R`에서 전체 비교를 다시 준비해야 하므로 추가 live 실행 전 팀에 알립니다.

## 실험 3 E5 정렬

먼저 선택한 장치에서 E5 test 점수를 만들고 이 명령의 경과시간을 저장합니다. 세 score 명령의 장치 마스크는 임베딩 프로세스에만 적용합니다.

각 score 명령은 점수 JSON 저장 후 null 점수 총개수·영향받은 논문 수·`_qid` 목록을 한 줄로 출력합니다. E5 test는 `test_e5_score.log`, BioLORD test는 `test_biolord_score.log`, validation은 `validation_biolord.log`에서 `null_scores=`를 찾으세요. 0개여도 요약을 출력하며, 빈 초록의 null도 집계합니다. 기존 원래 순서 평가·조기 종료·pruning 규칙은 유지합니다.

```bash
fresh "$R/test_e5.json" "$R/test_e5_score_wall_s.txt" "$R/test_e5_score.log"
check_embedding_context
/usr/bin/time -f '%e' -o "$R/test_e5_score_wall_s.txt" \
  env CUDA_VISIBLE_DEVICES="$EMBED_GPU_UUID" python semijoin_pruning.py score --model e5 --split test --n-articles "$N_ARTICLES" \
  --device "$EMBED_DEVICE" --output "$R/test_e5.json" > "$R/test_e5_score.log" 2>&1
```

LLM을 호출하기 전에 0부터 시작하는 test 행 번호, 양성 10편의 인덱스, 간 라벨 순서를 과거 기록과 대조합니다. 대조가 실패하면 먼저 입력을 확인합니다. 일치하더라도 과거 250편의 원문 전체와 데이터 revision까지 식별되는 것은 아닙니다.

```bash
python - "$R/test_e5.json" <<'PY'
import json, os, sys
from pathlib import Path
scores = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
rows = scores['articles']
assert scores['embedding_device'] == os.environ['EMBED_DEVICE']
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

3번과 같은 장치·GPU UUID에서 BioLORD test·validation 점수를 만듭니다. 아래 검사는 두 모델의 test 입력·조각·라벨 일치와 세 점수 파일의 장치가 선택값과 같은지 확인합니다. 검사를 통과한 뒤 **4번의 test LLM 판정을 보기 전에** validation의 정답과 점수로 5번의 두 임계값을 고정합니다. 불일치 시 기존 점수를 보존하고 입력·모델·장치 설정을 확인합니다.

```bash
test -s "$R/experiment3/summary.json"
fresh "$R/test_biolord.json" "$R/test_biolord_score_wall_s.txt" "$R/test_biolord_score.log"
check_embedding_context
/usr/bin/time -f '%e' -o "$R/test_biolord_score_wall_s.txt" \
  env CUDA_VISIBLE_DEVICES="$EMBED_GPU_UUID" python semijoin_pruning.py score --model biolord --split test --n-articles "$N_ARTICLES" \
  --device "$EMBED_DEVICE" --output "$R/test_biolord.json" > "$R/test_biolord_score.log" 2>&1
fresh "$R/validation_biolord.json" "$R/validation_biolord_wall_s.txt" "$R/validation_biolord.log"
check_embedding_context
/usr/bin/time -f '%e' -o "$R/validation_biolord_wall_s.txt" \
  env CUDA_VISIBLE_DEVICES="$EMBED_GPU_UUID" python semijoin_pruning.py score --model biolord --split validation --device "$EMBED_DEVICE" \
  --output "$R/validation_biolord.json" > "$R/validation_biolord.log" 2>&1
python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['R'])
read = lambda name: json.loads((root / name).read_text(encoding='utf-8'))
e5, bio, validation = map(read, ['test_e5.json', 'test_biolord.json', 'validation_biolord.json'])
for key in ['input_sha256', 'chunks_sha256', 'reactions']:
    assert e5[key] == bio[key], key
assert e5['scorer']['chunking'] == bio['scorer']['chunking']
assert all(scores['embedding_device'] == os.environ['EMBED_DEVICE'] for scores in (e5, bio, validation))
assert read('embedding-choice.json')['EMBED_DEVICE'] == os.environ['EMBED_DEVICE']
assert read('embedding-choice.json')['EMBED_GPU_UUID'] == os.environ['EMBED_GPU_UUID']
assert bio['scorer'] == validation['scorer']
assert bio['reactions'] == validation['reactions']
assert len(e5['articles']) == len(bio['articles']) == int(os.environ['N_ARTICLES'])
print('Shared abstracts/chunks/panel/device OK')
PY
fresh "$R/thresholds.json"
python semijoin_pruning.py calibrate "$R/validation_biolord.json" --output "$R/thresholds.json"
python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['R'])
validation = json.loads((root / 'validation_biolord.json').read_text(encoding='utf-8'))
calibration = json.loads((root / 'thresholds.json').read_text(encoding='utf-8'))
assert calibration['scorer'] == validation['scorer']
assert calibration['reactions'] == validation['reactions']
assert calibration['validation_input_sha256'] == validation['input_sha256']
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

위 합계는 두 명령을 각각 외부 타이머로 측정한 값입니다. **CPU/GPU 어느 장치든 test 임베딩 계산 비용을 포함합니다.** BioLORD test 점수는 한 번만 만들고, 각 방법을 독립적으로 실행할 때의 비용으로 4번과 5번에 1회분씩 반영합니다. Validation 보정, 서버 기동, 사전 준비·GPU 시험·연결 점검은 따로 기록합니다. 점수 계산 시 모델을 메모리에 올리는 시간과 온라인 메타데이터 조회는 측정에 포함되며, 파일을 실제로 추가 다운로드했다면 ‘네트워크·다운로드 포함’으로 표시합니다.

과거 전체 조인 7,282초와 고정 순서 조기 종료 1,896초는 실행 중인 GPU 서버에 연결한 Python 명령 시작부터 종료까지의 경과시간입니다. GPU 종류·수, 가중치·서버·동시성·캐시 조건이 확인될 때만 과거 대비 속도배율을 해석합니다. [과거 시간의 출처와 범위](HISTORICAL_70B_RESULTS.ko.md)

## 중단 시 재개

**중단된 경우에만** 이 단계를 사용합니다. 새 터미널에서 원래 Python 환경과 인증을 활성화하고 `source /실제/기존/R/settings.sh` 후 `cd "$BENCHMARK_DIR"`를 실행합니다. 이 파일에는 최종 임베딩 장치·GPU UUID도 보존돼 있습니다. 점수를 다시 만들어야 한다면 먼저 원래 호스트·GPU 할당과 `check_embedding_context`를 확인하고, 기존 점수·타이머를 덮지 않습니다. 원래 점수·출력 폴더·실험 번호를 사용하면 완료된 논문은 건너뜁니다. 아래는 첫 번째 3번 재개의 예입니다.

```bash
preflight experiment3_resume1
fresh "$R/experiment3_resume1_wall_s.txt" "$R/experiment3_resume1.log"
/usr/bin/time -f '%e' -o "$R/experiment3_resume1_wall_s.txt" \
  python semijoin_pruning.py run "$R/test_e5.json" --experiment 3 \
  --output-dir "$R/experiment3" > "$R/experiment3_resume1.log" 2>&1
```

4번은 원래 BioLORD 점수와 실험 번호를, 5번은 원래 calibration·policy까지 사용합니다. 재개할 때마다 사전 점검·로그·타이머에 새 이름을 붙입니다. 미완료 논문의 호출은 일부 빠질 수 있어 재개 시간을 더해 무중단 실측 시간으로 쓰지 않습니다. 전체 시간이 필요하면 보존한 점수를 사용해 새 출력 폴더에서 다시 실측합니다.

## 결과 백업

노드 예약이 끝나기 전에 **이번 `R` 디렉터리 전체**를 팀의 지속 저장소에 보관합니다. `comparison.json`의 품질 수치를 검토할 때에는 점수·임계값, 실험별 manifest·progress·articles·summary, replay 결과가 함께 필요합니다. 시간을 검토할 때에는 각 외부 타이머·실행 로그, 준비 로그, 서버 구성과 장비·패키지 기록, `embedding_inventory.log`·시험 로그·`embedding-choice.json`·`settings.sh`를 함께 사용합니다.
