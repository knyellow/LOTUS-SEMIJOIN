# BioDEX 실험 3·4·5 서버 실행 절차

Linux Bash에서 아래 순서로 실행합니다. **70B GPU 서비스에서 3번, 4번, 5번을 각각 한 번 실측합니다.** 임베딩은 사전 점검을 통과한 GPU 한 장을 우선 사용하고, 사용할 GPU가 없거나 시험에 실패하면 CPU를 사용합니다. **3번 시작 전에 한 번 결정해 네 번의 score 계산에 같은 장치를 유지합니다.** 3·4·5번은 점수 생성부터 각각 독립 실행하며, 5번은 validation 계산·threshold 보정 비용까지 포함합니다. 두 replay는 실측을 마친 뒤 별도로 계산합니다.

과거 실험 1·2번은 보고된 호출 수·논문 단위 F1·외부 시간을 별도로 인용합니다. 원본 요약이나 선택 CSV 없이 이 절차를 진행할 수 있습니다. 과거와 현재의 GPU·서버·캐시 조건을 확인하기 어려우면, 같은 환경에서 실행한 3~5번을 주 시간 비교로 사용합니다. [과거 결과와 출처](HISTORICAL_70B_RESULTS.ko.md) · [실험 설계와 지표](NEGATIVE_PRUNING_GUIDE.ko.md)

[서버 준비](#서버-준비) → [3번 E5](#실험-3-e5-정렬) → [4번 BioLORD](#실험-4-biolord-정렬) → [5번 pruning](#실험-5-pruning) → [품질 집계](#품질-집계) → [시간·순위 확인](#시간과-순위-확인) → [중단 시 재개](#중단-시-재개) · [백업](#결과-백업)

서버 AI에게 작업을 맡길 때는 [실행 요청 문서](SERVER_AI_EXPERIMENT_PROMPT.ko.md)를 먼저 전달합니다. 이 문서는 실제 명령을 제공하며, 실행 요청 문서는 수행 범위와 결과 보고 기준을 정합니다. 두 문서와 실행 코드는 같은 커밋을 사용합니다.

## 서버 준비

먼저 팀에서 사용할 수 있는 70B GPU 서비스와 실행 환경을 확인합니다. 과거 CloudLab 노드의 현재 접속 가능 여부는 미확인입니다. 서비스가 중지돼 있으면 해당 배포의 원래 기동 절차를 사용합니다.

| 항목 | 실행 조건 |
|---|---|
| GPU·모델 | 과거 기준은 V100 16GB 4장, `bartowski/Meta-Llama-3.1-70B-Instruct-GGUF`의 `Q4_K_M`입니다. 현재 서비스의 가중치·GPU·시작 설정을 대조해 차이를 기록합니다. |
| LLM 판정 | `PROMPT=paper`, 온도 0, 출력 최대 256토큰, `LLM_BATCH=8`, `EARLY_WORKERS=8`입니다. 시작 로그에서 슬롯당 context가 본문과 프롬프트·출력을 수용하는지 확인합니다. |
| 데이터 | BioDEX test 앞 250편, 본문 앞 8,000자, 간 라벨 10개를 사용합니다. Validation 전체는 5번의 임계값 보정에 사용합니다. |
| 점수 계산 | 아래 사전 시험 후 `EMBED_DEVICE=cpu` 또는 `cuda:0`을 고정합니다. E5 test 1회, BioLORD test 2회·validation 1회에 같은 장치·물리 GPU UUID를 사용하며 70B 설정은 유지합니다. |

**정책은 실행 전에 고정합니다.** Validation 양성 100% 보존(`conservative`)을 주 조건, 최소 95% 보존(`retain_95`)을 보조 조건으로 사용하고 실제 5번은 conservative 한 번만 실행합니다. 수치 threshold는 5번에서 validation만으로 계산하며, 4번 test 결과로 규칙을 바꾸지 않습니다.

팀원이 알려준 V100 8장은 서버에 설치된 수량이며, 과거 기록의 70B 사용 GPU 4장과 구분합니다. 실제 배정·사용 장수와 장당 VRAM을 확인합니다. RAM 48/96GB라는 설명만으로 GPU 여유를 판단하지 않습니다. 임베딩을 위해 70B를 다른 GPU에 재배치하지 않습니다.

### Python 환경 활성화

서버에서 팀의 LOTUS Conda/venv를 활성화한 뒤 저장소 루트에서 시작합니다. 이후 모든 명령은 같은 Bash 세션에서 순서대로 실행합니다.

먼저 `git status --short`에서 실험 코드·문서의 미커밋 변경과 영향을 줄 수 있는 미추적 파일을 확인합니다. 변경이 있으면 기존 작업을 보존하고 확인한 PR 커밋의 깨끗한 별도 체크아웃을 사용합니다. 아래 tracked 변경 검사가 실패하면 진행하지 않습니다. 커밋 SHA만 같다고 작업 폴더 내용까지 같지는 않으며, 이 준비 명령은 실험 중 반복하지 않습니다.

아래 `fetch/switch/pull`은 브랜치 작업 폴더 기준입니다. 이미 확인한 PR SHA의 깨끗한 detached worktree를 준비했다면 HEAD 일치를 대조한 뒤 이 세 줄을 생략하고 설치 단계로 진행합니다. 미커밋 변경 검사는 그대로 유지합니다. 다른 작업 폴더가 같은 브랜치를 사용 중이면 detached worktree에서 `git switch`가 거부될 수 있습니다.

```bash
set -euo pipefail
unset PYTHONOPTIMIZE
git diff --quiet HEAD --
git fetch origin
git switch feat/biolord-negative-pruning
git pull --ff-only
git diff --quiet HEAD --
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

`SERVER_CACHE_POLICY`는 기록용 값이며 실행기가 캐시를 초기화하지는 않습니다. 별도 초기화 절차를 적용했다면 실제 수행 여부와 시점을 기록하고, 기존 서비스를 유지했다면 초기 상태 미통제를 결과 해석에 반영합니다.

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

짧은 응답 시험은 출력 길이 인자를 포함한 요청의 수락만 확인합니다. 서버가 256토큰 상한을 실제로 적용하는지는 서버 설정·버전·로그로 별도 확인하고 미확인 사항은 기록합니다.

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

`cuda:0`은 원래 작업의 가시 목록에서 전체 UUID를 확인한 뒤, 후보 한 장만 보이게 한 **해당 임베딩 자식 프로세스의 번호**입니다. UUID를 식별할 수 없는 환경이면 CPU로 진행합니다. 70B 서버나 현재 셸의 GPU 가시성을 바꾸지 않습니다. CPU이면 임베딩 자식 프로세스에만 GPU를 숨깁니다. 선택 이후에는 네 score 계산의 장치·UUID·정밀도를 바꾸거나 점수 파일을 CPU/GPU 사이에 섞지 않습니다. 장치 선택 단계가 중단되어 시험 로그가 남았다면 보존하고 새 `R`에서 준비합니다. 본 점수 생성 중 GPU 오류가 나면 그대로 중단하고 원인을 확인합니다. 3번 이후 CPU로 바꿔야 한다면 기존 결과를 보존하고 새 `R`에서 전체 비교를 다시 준비해야 하므로 추가 live 실행 전 팀에 알립니다.

## 실험별 전체 실행과 시간

[run_semijoin_experiment.sh](run_semijoin_experiment.sh)가 아래 명령을 차례로 실행합니다. **기동된 70B 서버와 다운로드된 데이터·가중치 파일이 있는 조건**에서, 각 방법에 필요한 계산·프로세스 시작·입출력·묶음 내부 검증을 포함해 한 번의 외부 타이머로 측정합니다. 점수·단계 파일을 다른 실험에서 가져오지 않습니다.

| 실험 | 한 타이머에 포함되는 순서 |
|---|---|
| 3 | E5 test 점수 생성 → 기존 조기 종료 LLM 평가 |
| 4 | BioLORD test 점수 생성 → 기존 조기 종료 LLM 평가 |
| 5 | BioLORD validation 점수 생성 → 두 threshold 보정 → **자체 BioLORD test 점수 생성** → conservative pruning·LLM 평가 |

실행기는 Linux의 Bash·GNU `/usr/bin/time`·`setsid`를 사용합니다. 먼저 기존 출력 충돌을 검사하고 전용 workflow 폴더를 생성합니다. 70B 연결 점검과 GPU 할당 확인은 타이머 밖에서 하고, 측정 중 실패하면 뒤 단계를 실행하지 않습니다. 상태가 `complete`, 종료 코드가 0이며 모든 단계가 새로 완료된 경우에만 전체 시간을 사용합니다. 실행기는 `settings.sh`와 최초 장치 선택 기록도 대조합니다.

### 실험 3 E5 정렬

```bash
bash run_semijoin_experiment.sh 3 "$R"
cat "$R/experiment3_workflow/status.json"
```

`test_e5.json`을 만들고 3번을 실행합니다. `experiment3/summary.json`·`articles.json`이 생성돼야 합니다. 실행기는 LLM 평가 전에 250편·liver 라벨 순서·과거 양성 10개 행 번호·장치를 확인합니다. 행 번호가 같아도 과거 원문·revision까지 같다는 증명은 아닙니다.

### 실험 4 BioLORD 정렬

```bash
bash run_semijoin_experiment.sh 4 "$R"
cat "$R/experiment4_workflow/status.json"
```

**4번은 BioLORD test 점수 생성과 LLM 평가만 수행합니다.** Validation 임베딩과 threshold 계산은 하지 않습니다. `test_biolord.json`과 `experiment4/` 결과를 보존합니다.

### 실험 5 pruning

```bash
bash run_semijoin_experiment.sh 5 "$R"
cat "$R/experiment5_conservative_workflow/status.json"
```

5번은 `validation_biolord.json` → `thresholds.json` → `test_biolord_exp5.json` → `experiment5_conservative/`를 독립 생성합니다. **4번 점수나 LLM 판정을 실행 입력으로 사용하지 않습니다.** 기존 초록·128토큰 조각·모델·정밀도·장치 정책을 그대로 사용합니다. Validation 전체의 임베딩과 파일 저장, 보정 명령의 시작부터 종료까지도 5번 전체 시간에 포함됩니다.

5번 live는 conservative 한 번입니다. 보조 retain_95의 실제 시간은 미측정이며 아래 replay로 품질·논리 호출 수만 확인합니다. 모두 pruning이면 `articles.json`·`summary.json`은 생성되지만 `progress.jsonl`은 없어도 정상입니다.

| score 파일 | 생성 단계 | null 점수 요약 로그 |
|---|---|---|
| `test_e5.json` | 3번 | `test_e5_score.log` |
| `test_biolord.json` | 4번 | `test_biolord_score.log` |
| `validation_biolord.json` | 5번 | `validation_biolord.log` |
| `test_biolord_exp5.json` | 5번 | `test_biolord_exp5_score.log` |

각 로그의 `null_scores=`에서 null 총개수·영향 논문 수·행 번호를 읽습니다. 빈 초록의 null도 포함하고 0개여도 0/0/[]를 출력합니다. 원래 순서 평가·조기 종료·pruning 규칙은 바꾸지 않습니다.

## 실측 후 두 정책 replay

`calibrate`·`replay` CLI의 기존 출력 보호는 아직 보완하지 않았습니다. 현재 입력이 아닌 다른 기존 파일은 덮어쓸 수 있으므로 아래 `fresh`와 고정 출력 파일명을 유지합니다. 수동으로 경로를 바꾸면 **변경한 실제 출력 경로**가 없는지도 먼저 확인합니다.

**다음은 모든 실측 타이머 밖에서 수행하는 분석입니다.** 4번의 원본 점수·완료 판정에 5번에서 만든 두 threshold를 적용합니다. 추가 LLM 호출은 없으며 replay의 `lm_calls`는 정책을 적용했을 때의 예상 논리 호출 수입니다.

```bash
fresh "$R/experiment5_conservative_replay.json" "$R/experiment5_retain95_replay.json"
python semijoin_pruning.py replay "$R/test_biolord.json" \
  --calibration "$R/thresholds.json" --policy conservative \
  --source-dir "$R/experiment4" --output "$R/experiment5_conservative_replay.json"
python semijoin_pruning.py replay "$R/test_biolord.json" \
  --calibration "$R/thresholds.json" --policy retain_95 \
  --source-dir "$R/experiment4" --output "$R/experiment5_retain95_replay.json"
```

## 품질 집계

실측마다 **그 실행이 만든 원본 점수 파일**을 짝지어 줍니다. 5번에는 `test_biolord_exp5.json`, 4번과 그 replay에는 `test_biolord.json`을 사용합니다. 집계 중 LLM 호출은 없습니다.

```bash
fresh "$R/comparison.json"
python semijoin_pruning.py report --scores "$R/test_biolord.json" \
  --run "$R/test_e5.json" "$R/experiment3" \
  --run "$R/test_biolord.json" "$R/experiment4" \
  --run "$R/test_biolord_exp5.json" "$R/experiment5_conservative" \
  --replay "$R/experiment5_conservative_replay.json" \
  --replay "$R/experiment5_retain95_replay.json" \
  --output "$R/comparison.json"
```

보고서는 3 live·2 replay 행이며 `missing_experiments=[1,2]`, `vs_join_full=null`이 정상입니다. 과거 선택 CSV가 없어 1번과의 논문별 일치도는 계산하지 않습니다. 논문 단위 TP·FP·FN·precision·recall·F1은 `results[].vs_labels`에서 읽습니다.

각 live/replay는 자기 원본 점수 **전체 해시**를 검증합니다. 4·5번 간 비교에서는 독립 재계산으로 달라지는 `embedding_time_s`만 제외하고 입력·모델·조각·라벨·장치·실제 점수·gold가 같아야 합니다. 실제 점수가 미세하게라도 다르면 보고서가 거부합니다. 두 결과를 보존해 원인과 정렬·두 threshold의 gate 결정을 확인하세요. 허용 오차를 임의로 넣거나 5번 파일을 4번 파일로 바꿔 통과시키지 않습니다.

## 시간과 순위 확인

주 시간은 **`실험명_workflow/total_wall_s.txt`**입니다. 각 단계의 `*_wall_s.txt`는 보조 정보입니다. 이들을 다시 더해 주 시간을 만들거나, validation·보정이 빠진 내부 추정 시간을 5번 전체 시간으로 사용하지 않습니다.

```bash
python - <<'PY'
import json, math, os
from pathlib import Path
from semijoin_pruning import digest, load_scores
root = Path(os.environ['R'])
read = lambda path: json.loads(path.read_text(encoding='utf-8'))
names = ['experiment3', 'experiment4', 'experiment5_conservative']
phases = [
    ['test_e5_score', 'experiment3'],
    ['test_biolord_score', 'experiment4'],
    ['validation_biolord', 'calibration', 'test_biolord_exp5_score', 'experiment5_conservative'],
]
completed = []
for number, name, stages in zip([3, 4, 5], names, phases):
    workflow = root / f'{name}_workflow'
    status = read(workflow / 'status.json') if (workflow / 'status.json').exists() else {}
    summary_path = root / name / 'summary.json'
    if status.get('status') != 'complete' or status.get('exit_code') != 0 or not summary_path.exists():
        print(name, 'overall wall: unavailable (failed, interrupted, or no completed workflow)')
        continue
    summary = read(summary_path)
    if summary['resumed']:
        print(name, 'overall wall: unavailable (resumed)')
        continue
    assert status['experiment'] == number
    try:
        total = float((workflow / 'total_wall_s.txt').read_text().strip())
        stage_times = {p: float((root / f'{p}_wall_s.txt').read_text().strip()) for p in stages}
    except (OSError, ValueError):
        print(name, 'overall wall: unavailable (missing or invalid timer)')
        continue
    assert all(math.isfinite(t) and t >= 0 for t in [total, *stage_times.values()])
    assert total + 0.1 >= sum(stage_times.values()), 'Outer timer may have been overwritten'
    assert stage_times[name] + 0.1 >= summary['invocation_time_s']
    rows = read(root / name / 'articles.json')
    assert [r['_qid'] for r in rows if r['gold_positive']] == [29, 119, 134, 143, 148, 153, 158, 167, 198, 223]
    ranks = [r['first_gold_witness_rank'] for r in rows if r['gold_positive']]
    print(name, 'calls', summary['lm_calls'], 'F1', summary['f1'], 'recall', summary['recall'])
    print('mean first gold witness rank', sum(ranks) / len(ranks) if ranks else None)
    print('overall wall seconds', total, 'stage wall seconds', stage_times)
    completed.append(name)
if completed == names:
    ms = [read(root / name / 'manifest.json') for name in names]
    assert all(m['model'] == os.environ['LLM_MODEL'] and m['workers'] == 8 and m['llm_batch'] == 8 for m in ms)
    assert all(m['max_tokens'] == 256 and m['temperature'] == 0 for m in ms)
    for key in ['model', 'instruction', 'max_tokens', 'temperature', 'workers', 'llm_batch', 'endpoint_sha256', 'llm_server_id']:
        assert len({json.dumps(m[key], sort_keys=True) for m in ms}) == 1, key
    scores = [load_scores(root / p) for p in ['test_e5.json', 'test_biolord.json', 'test_biolord_exp5.json']]
    validation = load_scores(root / 'validation_biolord.json')
    choice = read(root / 'embedding-choice.json')
    for key in ['EMBED_DEVICE', 'EMBED_GPU_UUID', 'EMBED_HOST', 'EMBED_ORIGINAL_VISIBILITY']:
        assert choice[key] == os.environ[key], key
    assert all(s['embedding_device'] == os.environ['EMBED_DEVICE'] for s in [*scores, validation])
    for key in ['input_sha256', 'chunks_sha256', 'reactions']:
        assert len({digest(s[key]) for s in scores}) == 1, key
    assert len({digest(s['scorer']['chunking']) for s in scores}) == 1
    assert {k: v for k, v in scores[1].items() if k != 'embedding_time_s'} == {k: v for k, v in scores[2].items() if k != 'embedding_time_s'}
    calibration = read(root / 'thresholds.json')
    assert validation['scorer'] == scores[2]['scorer'] == calibration['scorer']
    assert validation['reactions'] == scores[2]['reactions'] == calibration['reactions']
    assert validation['input_sha256'] == calibration['validation_input_sha256']
    print('Cross-experiment settings and score checks passed.')
else:
    print('Cross-experiment check: unavailable until all three fresh workflows complete.')
PY
```

5번 주 시간에는 validation 전체 임베딩·threshold 보정·자체 test 점수 생성·LLM 평가가 모두 포함됩니다. 전체 묶음에는 내부 검증·프로세스·입출력 비용도 있어 단계 시간의 합과 정확히 같지는 않습니다. Test-only 시간이나 사전 보정 비용을 제외한 반복 운영 비용을 보조 분석한다면 이름과 제외 범위를 별도로 명시합니다.

최초 파일 다운로드·70B 서버 기동·GPU/CPU 시험·사전 연결 점검·실측 후 replay와 공통 보고서는 주 시간 밖입니다. 모든 방법에 같은 준비·cache 조건을 적용합니다. 측정 중 실제 추가 다운로드가 발생했다면 해당 시간에 포함됐다고 표시합니다. 과거 1·2번의 외부 시간은 그대로 인용하되, 장비·가중치·서버·cache 동일성을 확인하지 못하면 참고값으로 두고 현재 3~5번을 주 시간 비교로 사용합니다.

## 중단 시 재개

실행기가 실패하거나 중단되면 원본 workflow 상태·타이머·점수·진행 기록을 보존합니다. **일부 단계부터 재개한 시간을 처음부터 완료한 전체 시간으로 쓰지 않습니다.** LLM `summary.resumed=false`여도 앞선 score/calibrate 단계를 재사용했다면 새로운 전체 시간으로 인정할 수 없습니다. 완전한 새 실측은 새 출력 위치에서 모든 필수 단계를 다시 수행해야 하므로 추가 live 실행 전 팀에 보고합니다.

실행기에 HUP·INT·TERM이 전달되면 해당 측정 작업 그룹에도 종료를 전달하고 실패 상태를 기록합니다. 공유 70B 서버는 종료 대상이 아닙니다. SIGKILL·노드 종료는 상태를 기록할 수 없어 `running`이 남을 수 있으며 성공으로 해석하지 않습니다. 다시 제출하기 전에 기존 작업이 종료됐는지 확인합니다.

완료되지 않은 **LLM 평가만 복구**할 때는 기존 환경과 인증을 활성화하고 `source /실제/기존/R/settings.sh`, `cd "$BENCHMARK_DIR"` 후 아래와 같이 실행합니다. 이 작업은 workflow를 complete로 바꾸지 않으며 전체 시간은 미측정으로 남습니다.

```bash
preflight experiment3_resume1
fresh "$R/experiment3_resume1_wall_s.txt" "$R/experiment3_resume1.log"
/usr/bin/time -f '%e' -o "$R/experiment3_resume1_wall_s.txt" \
  python semijoin_pruning.py run "$R/test_e5.json" --experiment 3 \
  --output-dir "$R/experiment3" > "$R/experiment3_resume1.log" 2>&1
```

4번은 원래 `test_biolord.json`, 5번은 **`test_biolord_exp5.json`과 원래 `thresholds.json`·conservative 정책**을 사용합니다. 같은 설정만 재개하며 점수·최초 타이머를 덮지 않습니다. 점수를 다시 만들어야 하면 원래 호스트·GPU 할당을 재확인합니다. 완료된 workflow를 재호출하면 기존 파일을 보호하기 위해 거부합니다.

## 결과 백업

예약 종료 전에 **이번 `R` 전체**를 지속 저장소에 보관하고 파일 목록·크기·SHA-256으로 일치를 확인합니다. 네 점수 JSON, threshold, live 세 폴더, replay 두 파일, comparison, 단계 타이머/로그, 각 workflow의 `status.json`·`total_wall_s.txt`·`execution.log`, 장치 선택·시험·환경·서버 기록을 모두 포함합니다. 인증 정보와 모델 가중치 파일 자체는 결과 백업에 넣지 않습니다.
