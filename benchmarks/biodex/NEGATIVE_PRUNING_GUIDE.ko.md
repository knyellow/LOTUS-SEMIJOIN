# BioDEX semi-join: 코드와 설계 설명

이 문서는 **어떤 코드를 바꿨고 왜 그렇게 구현했는지** 설명합니다. 서버에서 실험을 실행하려면 [실행 가이드](NEGATIVE_PRUNING_RUNBOOK.ko.md)를, 기존 1·2번 수치의 근거를 확인하려면 [과거 70B 결과](HISTORICAL_70B_RESULTS.ko.md)를 읽으세요.

## 1. 해결하려는 문제

Semi-join은 각 논문에 대해 **조건을 만족하는 라벨이 하나라도 있는지** 확인하는 연산입니다. 매칭을 하나 찾으면 같은 논문의 나머지 라벨은 평가할 필요가 없습니다. 이 프로젝트는 기존 조기 종료 실행기에 임베딩 점수를 연결해, 매칭을 더 일찍 찾거나 관련성이 낮은 논문을 건너뛸 수 있는지 확인합니다.

| 실험 | 변경하는 것 | 확인할 효과 |
|---|---|---|
| **3: E5 정렬** | E5 점수가 높은 라벨부터 LLM에 질문합니다. | 기존 순서보다 호출을 줄이는지 |
| **4: BioLORD 정렬** | 같은 초록 조각을 BioLORD로 점수화합니다. | E5보다 witness를 앞에서 찾는지 |
| **5: BioLORD + pruning** | 모든 라벨에 점수가 낮은 논문은 평가를 생략합니다. | 추가 절감과 정답 누락의 교환 |

**새 실측은 3→4→5 각 1회, 총 3회입니다.** 기존 70B 1·2번 수치는 별도 표로 인용합니다. 원본 선택 CSV가 없어 새 결과와의 `vs_join_full`은 계산하지 않습니다. 두 threshold의 replay는 추가 LLM 호출이 없습니다.

3·4번은 같은 pair 판정이 주어지면 순서만 바꿔도 선택 결과가 같습니다. 5번은 논문을 생략하므로 결과를 바꿀 수 있는 근사 최적화입니다. 실제 live 실행은 LLM 판정 변동이 있을 수 있어 품질도 함께 측정합니다.

## 2. 확정한 설정과 이유

| 항목 | 이번 설정 | 이유 |
|---|---|---|
| LLM | 70B Instruct Q4_K_M, GPU, batch·worker 8 | 재사용하는 1·2번의 보고된 모델·판정 조건에 맞춥니다. |
| 데이터·질문 | BioDEX test 앞 250편, liver 10개 라벨, `paper` | 같은 질문과 논문 범위에서 비교합니다. |
| Proxy 입력 | **Abstract 전체** | 초록의 뒷부분도 순서·생략 판단에 활용합니다. |
| LLM 입력 | **`fulltext_processed[:8000]`** | 기존 LLM 판정 입력을 유지합니다. 단위는 문자입니다. |
| 임베딩 장치 | **사전 시험을 통과한 GPU 한 장, 사용할 수 없으면 CPU** | 70B 배정을 유지하면서 임베딩을 가속합니다. 3번 전에 선택하고 세 점수 생성의 장치를 통일합니다. |
| 조각·점수 | 공통 128토큰 이내, 겹침 없음, 라벨별 최대 cosine similarity | 같은 텍스트 조각에서 모델을 비교하고 어느 조각의 관련 근거든 포착합니다. |
| Threshold | Validation gold 양성 100% 보존이 주 조건, 최소 95%가 보조 조건 | 임의의 cutoff 대신 검증 데이터의 정답으로 비용·손실 조건을 정합니다. |

**Validation 보존율은 test 무누락이나 test recall을 보장하지 않습니다.** Test에서 버린 양성과 최종 품질을 확인합니다. 과거와 현재의 장비·빌드·cache 조건을 확인하지 못하면 과거 시간은 참고값으로 표시합니다.

<details>
<summary>입력·모델 비교에 관한 세부 근거</summary>

- E5는 기본적으로 512토큰까지 받을 수 있지만, 이번에는 BioLORD와 같은 원문 조각을 사용합니다. 각 모델을 별도로 최적화한 최고 성능 비교가 아니라 **같은 초록 조각에서의 모델 비교**입니다.
- 한 조각의 의미가 라벨과 가깝다면 그 논문 안에서 해당 라벨을 먼저 평가하려고 최대 점수를 사용합니다. 긴 초록은 우연히 높은 최대값을 얻을 수 있으므로 threshold는 같은 조각 정책과 라벨 패널에서만 적용합니다.
- 겹침을 두지 않아 중복 임베딩을 줄입니다. 경계에서 문맥이 끊길 수 있다는 한계는 남습니다.
- GPU 수나 시스템 RAM만으로 사용 가능 여부를 판단하지 않습니다. 실제 score 실행 호스트에서 배정받은 GPU를 확인하고 두 모델을 별도 프로세스로 시험합니다. 사용할 수 없으면 CPU로 정하며, E5 test와 BioLORD test·validation 모두 선택한 같은 장치·GPU UUID를 사용합니다. [실행 전 장치 점검](NEGATIVE_PRUNING_RUNBOOK.ko.md#임베딩-장치-점검선택)
- 가중치·입력·정밀도·정규화는 장치 선택과 무관하게 유지합니다. CPU/GPU의 부동소수점 계산은 완전히 같지 않을 수 있으므로 점수·validation 보정을 섞지 않습니다. [PyTorch 수치 정확도 설명](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)
- `paper`는 조건이 논문에 기술됐는지를 묻는 질문입니다. 실제 약물 부작용 여부를 묻는 `adr`로 바꾸지 않습니다. Test 정답은 순서·생략·LLM 입력에 넣지 않고 평가에만 사용합니다.
- 고정된 test 자료에서 abstract 전체가 LLM 입력 8,000자 안에도 포함되는지는 [별도 증거 PR](https://github.com/knyellow/LOTUS-SEMIJOIN/pull/2)에서 검사합니다. 이 검사는 과거 노드의 데이터 revision이나 실제 서버의 추가 잘림까지 확인하지는 않습니다.

</details>

## 3. 변경 파일과 호출 경로

필요한 처리는 벤치마크 코드에 두고, 임베딩과 논문–라벨 판정은 기존 LOTUS 코드를 재사용했습니다.

| 파일 | 역할과 변경 이유 |
|---|---|
| [semijoin_pruning.py](semijoin_pruning.py) | 점수 생성·보정·실험 실행·replay·집계를 연결합니다. |
| [semijoin_baseline.py](semijoin_baseline.py) | 기존 설정·로더·조기 종료 실행기에 선택적 기능만 추가했습니다. |
| [sentence_transformers_rm.py](../../lotus/models/sentence_transformers_rm.py) | 기존 임베딩 래퍼를 그대로 사용합니다. |
| [sem_join.py](../../lotus/sem_ops/sem_join.py) | 기존 pair 판정 경로를 그대로 사용합니다. |
| [test_semijoin_pruning.py](../../tests/test_semijoin_pruning.py) | 경계값·결과 혼용·재개·집계 등 실패 가능한 동작을 검증합니다. |
| [tests.yml](../../.github/workflows/tests.yml) | 변경 파일을 Ruff와 기존 settings 회귀 검사에 연결합니다. |

```text
score      load_articles(include_abstract=True)
           → abstract_chunks → SentenceTransformersRM → 조각별 점수의 max
calibrate  validation 점수·gold → calibrate → 두 threshold
run        입력·설정 검증 → gate·ordering → 기존 run_early_stop
           → apply_gate → summarize → 결과 저장
replay     completed_run 검증 → 저장된 판정에 gate 적용 → 새 품질·호출 수
report     완료 결과·선택적 과거 CSV 검증 → comparison_prf → comparison.json
```

`semijoin_baseline.py`의 변경은 네 가지입니다. `lm_config()`가 기존 LLM 설정을 반환하고, `configure(with_retriever=False)`가 점수가 이미 있는 실행의 불필요한 E5 로딩을 생략합니다. `load_articles()`는 split·abstract 선택을 받고, `run_early_stop()`은 선택적 `reaction_orders`를 받습니다. 기본 호출의 본문·라벨 순서·판정 방식은 유지합니다. LOTUS semantic operator 자체는 수정하지 않습니다.

## 4. 점수에서 LLM 판정까지

### 4.1 입력을 준비하고 같은 초록 조각을 만듭니다

`load_articles()`는 지정 split 앞 N편에 원래 행 번호 `_qid`를 부여합니다. Proxy용 `abstract`, LLM용 `patient_description`, 평가용 `reactions_list`의 역할을 구분합니다. `_qid`는 PMID가 아니므로 숫자만 같다고 과거와 같은 논문임을 입증할 수는 없습니다.

`abstract_chunks()`는 두 tokenizer에서 모두 제한 안에 들어가는 원문 구간을 만듭니다. 길이에는 special token과 E5 문서 접두어도 포함합니다. 가능한 경우 단어 경계를 택하며, 조각을 이어 붙이면 초록 원문이 복원됩니다.

<details>
<summary>입력·조각 처리의 작은 코드 단위</summary>

```python
patient_description = fulltext_processed[:8000]  # 기존 LLM 입력
chunks = abstract_chunks(abstract, tokenizers)    # Proxy는 초록 전체
```

`fits_chunk()`는 E5에는 `passage: `를 붙이고 BioLORD에는 붙이지 않은 길이를 각각 검사합니다. 두 tokenizer의 조건을 모두 만족해야 합니다. 공백뿐인 초록은 조각을 만들지 않습니다. 한 글자도 제한에 들어가지 않으면 조용히 자르지 않고 오류를 냅니다.

Gold 라벨과 표적 라벨이 하나라도 겹치는 논문을 article-level 정답 양성으로 둡니다. 이 정보는 점수 파일에 평가용으로 저장되며 LLM에는 본문과 현재 후보 라벨만 전달합니다.

</details>

### 4.2 조각과 라벨을 임베딩해 논문별 점수를 구합니다

`make_scores()`는 한 번에 선택한 임베딩 모델 하나를 로드합니다. 다른 모델의 tokenizer는 공통 길이 검사에 사용합니다. E5는 문서에 `passage: `, 라벨에 `query: `를 붙이며 BioLORD에는 접두어를 붙이지 않습니다.

기존 `score --device`가 LOTUS 래퍼를 거쳐 실제 SentenceTransformer 장치에 전달되므로 GPU 전환을 위한 Python 수정은 필요하지 않습니다. GPU에서는 임베딩 forward를 실행하며, 토크나이징·초록 분할·벡터를 CPU로 가져온 뒤의 유사도·max 집계는 기존처럼 CPU입니다. CPU 복사가 완료된 후 내부 시간을 기록하고, score 프로세스 종료로 해당 임베딩 GPU 메모리를 반환합니다. 70B 서버의 상주 메모리는 이와 별개입니다.

정규화한 벡터의 내적으로 cosine similarity를 계산하고, 같은 논문·라벨의 조각 점수 중 최대를 사용합니다. 예를 들어 한 라벨의 조각 점수가 0.2와 0.8이면 논문 점수는 0.8입니다. 이 점수는 정답 판정이 아니라 평가 순서·생략 여부를 정하는 proxy입니다.

<details>
<summary>점수 계산과 무효 임베딩 처리</summary>

```python
chunk_scores = vectors @ labels.T
similarities[index] = chunk_scores[offset:stop].max(axis=0)
```

`offset:stop`은 현재 논문의 조각 범위이고 `axis=0`은 같은 라벨의 최대를 뜻합니다. 실제 코드는 이 집계 전에 NaN·무한대·영벡터를 검사합니다. 중요한 조각의 근거가 빠질 수 있으므로 한 조각이라도 무효이면 해당 논문의 proxy를 신뢰하지 않습니다. 빈 초록·무효 점수는 원래 라벨 순서로 평가하고 pruning하지 않습니다. JSON의 `null`은 사용할 수 없는 점수입니다.

점수 JSON 저장에 성공하면 `null_scores=20, affected_articles=2, qids=[7, 18]` 형태로 한 줄을 출력합니다. `null_scores`는 라벨별 사용 불가 점수의 총개수, `affected_articles`와 `qids`는 그 점수가 하나라도 있는 논문 수와 행 번호입니다. 빈 초록에서 생긴 null도 포함하며, 0개이면 `null_scores=0, affected_articles=0, qids=[]`를 출력합니다. 점수 파일 형식과 무효 점수 처리 규칙은 바꾸지 않습니다.

모델 revision·정규화·접두어·조각 정책·tokenizer 내용을 기록합니다. `input_sha256`은 논문 ID·본문·초록·정답을, `chunks_sha256`은 원문 조각 목록을 묶습니다. Tokenizer 해시에서는 배칭 중 변하는 padding·truncation 상태를 제외합니다.

</details>

### 4.3 순서를 정하고, 5번에서만 논문 전체를 거릅니다

`ordering()`은 라벨을 점수 내림차순으로 정렬하고 동점은 원래 순서를 유지합니다. 3·4번은 모든 논문을 평가하고, 5번은 최대 라벨 점수가 threshold보다 **작은 논문만** 생략합니다. 같은 점수이면 유지합니다.

| 최대 점수와 threshold의 관계 | 동작 |
|---|---|
| 최대 점수 > threshold | 높은 라벨부터 LLM 평가 |
| 최대 점수 = threshold | 평가 |
| 최대 점수 < threshold | 논문의 모든 LLM 평가 생략 |
| 초록·점수 사용 불가 | 생략하지 않고 원래 순서로 평가 |

`run_locked()`는 남긴 논문과 정렬 순서를 기존 실행기에 전달합니다. 첫 True에서 논문을 선택하고 종료하며, 모두 False이면 제외합니다. 논문끼리는 worker 8개로 병렬 처리하고 한 논문의 라벨은 순차 평가합니다.

<details>
<summary>기존 LOTUS 판정까지 이어지는 호출</summary>

```python
baseline.run_early_stop(
    articles[articles['_qid'].isin(kept)],
    scores['reactions'], lm, str(progress_path),
    reaction_orders=orders,
)
```

```text
run_early_stop → pairwise_join = lotus.sem_ops.sem_join.sem_join
              → sem_filter → filter_formatter
              → CountingLM → LM → LiteLLM
              → filter_postprocess → join_results
```

핵심 반복은 pair 판정 후 호출 수를 하나 증가시키고, 수락한 pair가 있으면 `break`하는 것입니다. 기존 paper·adr 문자열과 temperature 0·출력 256 설정을 유지합니다. 해석 실패 시 기본값 True인 parser 동작도 유지하므로 witness는 **현재 LLM/parser가 수락한 첫 pair**이며 gold 정답과 다를 수 있습니다.

</details>

### 4.4 Threshold는 validation에서 먼저 고정합니다

`calibrate_file()`은 전체 validation의 BioLORD 점수와 gold 양성을 사용합니다. 주 조건 `conservative`는 gold 양성 전부, 보조 조건 `retain_95`는 최소 95%를 남기는 경계 점수를 구합니다. LLM 호출은 없습니다. 두 값은 4번 test LLM 결과를 보기 전에 고정하고 test 결과로 다시 조정하지 않습니다.

<details>
<summary>100%·95%를 정수 개수로 처리하는 이유</summary>

```python
allowed_misses = len(positives) - math.ceil(target * len(positives))
threshold = finite_positive_scores[
    min(allowed_misses, len(finite_positive_scores) - 1)
]
```

양성이 10편이면 95% 정책도 한 편을 버릴 수 없고, 20편이면 최대 한 편을 버릴 수 있습니다. 동점·무효 점수 때문에 실제 누락은 더 적을 수 있어 두 threshold가 같아져도 정상입니다. 무효 점수는 항상 유지하고, 유효 점수 양성이 하나도 없으면 보정을 중단합니다. Calibration과 test의 모델·라벨·조각 정책이 달라지면 적용하지 않습니다.

</details>

### 4.5 Replay와 report는 저장된 결과를 검증한 뒤 계산합니다

`replay`는 4번에서 저장한 판정에 두 gate를 적용합니다. 정책별 품질·논리적 호출 수는 계산할 수 있지만, 실제 추가 LLM 호출은 0회이고 정책의 실측 시간은 없습니다.

`report`는 선택 논문 ID를 같은 gold ID와 비교해 P/R/F1을 다시 계산합니다. 기존 `baseline.prf()`를 재사용해 반올림을 통일하고 정수 TP·FP·FN을 붙입니다. 저장된 실행 설정을 비교하므로 집계 컴퓨터의 모델 기본값으로 과거 설정을 덮어 해석하지 않습니다.

이번 입력에는 과거 원본을 넣지 않아 `missing_experiments=[1,2]`, `vs_join_full=null`이 정상입니다. 공통 보고서는 품질·편수·호출 수를 집계하며 시간과 witness 순위는 원본 결과에서 읽습니다.

## 5. 저장·재개·검증이 필요한 이유

| 파일 | 담는 내용 |
|---|---|
| 점수 JSON | 모델·장치·조각·입력 식별자, 논문별 점수·gold |
| `embedding-choice.json`·시험 로그 | 최종 CPU/GPU 선택·물리 GPU UUID·호스트·선택 이유와 두 모델의 실행 점검 |
| `manifest.json` | 실험·점수 해시·threshold·LLM·endpoint·서버 기록 식별자 |
| `progress.jsonl` | 평가 완료 논문의 `_qid`, `matched`, `lm_calls`, `time_s` |
| `articles.json` | 생략 논문을 포함한 전체 결과·gold·순위 |
| `summary.json` | 실측 실행의 품질·편수·호출 수·내부 시간 |
| Replay JSON | 원본 manifest, 새 `summary`, 새 `articles` 배열 |
| `comparison.json` | `results` 배열과 공통 품질·출처·누락 실험 정보 |

모델·입력·정렬이 바뀐 기록을 섞으면 호출 수와 품질의 의미가 달라집니다. 따라서 점수는 덮어쓰지 않고, 같은 설정만 같은 폴더에서 재개하며, 새 시간 측정은 새 출력에서 시작합니다. 정확한 명령은 [중단 시 재개](NEGATIVE_PRUNING_RUNBOOK.ko.md#중단-시-재개)를 참고하세요.

<details>
<summary>파일 보호와 완료 기록 검증의 세부 동작</summary>

- 출력 잠금은 같은 폴더의 동시 writer를 막습니다. 중단된 JSONL은 마지막 미완성 꼬리만 복구하며 잘못된 완료 기록은 거부합니다.
- `summary.manifest_sha256`은 설정과 결과를 연결합니다. 완료 재호출은 최초 요약·시간을 보존하고, 동일 기록으로 누락된 `articles.json`만 복구할 수 있습니다.
- `completed_run()`은 replay와 report가 공유합니다. 원본 점수, 전체 평가 기록, 논문별 결과, 요약, manifest를 대조하며 보고 과정에서는 원본을 복구·수정하지 않습니다.
- 공통 집계는 같은 입력·gold·라벨·조각·LLM 설정을 요구합니다. 4·5번은 정확히 같은 BioLORD 점수 파일이어야 합니다. Replay에는 검증된 원본 live run도 필요합니다.
- `LLM_SERVER_ID`와 해시는 저장 기록의 혼용을 검출합니다. 실제 가중치 파일·과거 서버의 동일성을 자동 입증하는 기능은 아닙니다.
- 점수의 `embedding_device`는 요청 장치 문자열이고 `cuda:0`만으로 물리 GPU를 구별할 수 없습니다. CLI 보정·집계는 세 점수의 CPU/GPU 혼용을 자동 차단하지 않으므로 실행 문서에서 장치·UUID를 고정하고 선택 기록 및 세 파일을 대조합니다. `settings.sh`는 최종 선택 후 재개용 설정을 보존합니다.
- `check-baselines`와 `run --baseline-summary`는 과거 요약에 기록된 설정을 대조하는 선택 기능입니다. `report --baseline-summary`는 선택 CSV까지 읽어 ID·편수·기존 품질 값과 대조합니다. 지정한 원본이 누락되거나 충돌하면 미측정으로 숨기지 않고 거부합니다. 이번 실행의 필수 입력은 아닙니다.

</details>

## 6. 지표를 읽고 비교하는 방법

### 공통 품질과 호출 수

| 지표 | 실측 `summary.json` | 공통 `comparison.json` | 의미 |
|---|---|---|---|
| 전체·gold 양성·선택 논문 수 | `n_articles`, `n_positive`, `n_selected` | `results[]`의 같은 이름 | 중복을 제거한 논문 단위 개수 |
| LLM 호출 수 | `lm_calls` | `results[].lm_calls`, `lm_calls_kind` | 논리적 pair 판정 수. HTTP 재시도·토큰 수와 다릅니다. |
| 정답 기준 P/R/F1 | `precision`, `recall`, `f1`, `tp`, `fp`, `fn` | `results[].vs_labels` | 같은 전체 test 정답 기준 |
| Full Join 기준 일치도 | 실측 summary에 자동 출력 없음 | `results[].vs_join_full` | 이번에는 과거 선택 ID가 없어 `null` |

품질은 **논문 선택의 정확도**입니다. Gold 양성 논문을 선택하면 TP이며, LLM이 지목한 개별 라벨의 정확도는 아닙니다. 5번이 생략한 양성도 전체 분모에서 FN으로 남습니다. 과거 1·2번은 보고된 호출 수·F1·외부 시간만 별도 표에 인용합니다.

### Witness 순위와 추가 손실

| 지표 | 읽을 값 | 비교할 대상 |
|---|---|---|
| 첫 LLM witness | `articles.json`의 `matched != null`인 행의 `lm_calls` | 비교할 방법 모두 witness가 있는 같은 ID 집합. 편수도 표시합니다. |
| 저장된 첫 LLM witness 평균 | `summary.mean_first_llm_witness_rank` | 각 방법의 선택 집합 평균. 대상이 다르면 직접 개선으로 해석하지 않습니다. |
| 첫 gold witness | `articles.json`의 `first_gold_witness_rank` | 같은 gold 양성 10편 전체에서 3·4번 비교 |
| Gate 양성 손실 | `n_positive_pruned`, `positive_retention` | 생략한 gold 양성 수와 gate 보존율. 최종 LLM recall이 아닙니다. |
| 4번 대비 추가 FN | `{4번이 선택한 gold ID} - {5번이 선택한 gold ID}`의 크기 | Replay로 gate 손실을 분리하고 별도 live 차이도 확인합니다. |

Witness가 없으면 순위도 없습니다. 모든 라벨이 False인 논문의 K회 호출과 pruning의 0회를 순위에 넣지 않습니다. 4·5번은 유지한 논문의 순서가 같으므로 5번 평균 변화는 선택 대상 변화일 수 있습니다. Replay 평균은 JSON의 `summary.mean_first_llm_witness_rank`, 논문별 값은 `articles` 배열에서 읽습니다.

<details>
<summary>추가 FN 계산 예시와 과거 원본을 복구한 경우</summary>

```python
tp4 = {r['_qid'] for r in rows4 if r['gold_positive'] and r['matched'] is not None}
tp5 = {r['_qid'] for r in rows5 if r['gold_positive'] and r['matched'] is not None}
additional_fn = len(tp4 - tp5)
```

`n_positive_pruned`에는 4번도 놓쳤던 양성이 포함될 수 있어 위 추가 FN과 다릅니다. F1이 올라도 오탐 제거 효과일 수 있으므로 LLM 자체의 판정 능력이 좋아졌다고 해석하지 않습니다.

과거 원본을 복구하면 1번은 `runs.join_full`, 2번은 `runs.early_stop`에서 `lm_calls`, `n_gold`, `vs_labels.n_pred`, `vs_labels.{precision,recall,f1}`를 읽습니다. 새 값은 각각 `lm_calls`, `n_positive`, `n_selected`, 품질 필드에 대응합니다. 1번 `join_rows`는 매칭 pair 수라 선택 논문 수와 비교하면 안 됩니다. 선택 수는 `*_semi.csv`의 고유 `_qid`로, 1번은 pair CSV를 투영해서도 계산할 수 있습니다.

과거 1번 가상 순위는 `_qid`별 매칭 라벨의 `reactions.index(reaction)+1` 최솟값입니다. CSV 행 순서를 쓰지 않습니다. 2번은 `early_stop_progress.jsonl`에서 수락한 논문의 `lm_calls`입니다. 현재는 원본이 없어 과거와의 같은-ID 순위 비교를 하지 않습니다.

</details>

### 실행시간

시간은 **test score 명령의 외부 wall time + live run 명령의 외부 wall time**으로 비교합니다. CPU/GPU 어느 쪽이든 임베딩 계산을 포함하고, GPU 사전 시험은 별도 준비 비용입니다. 4·5번에는 공유한 BioLORD test score 시간 1회분을 각각 포함합니다. 새 실행끼리는 같은 CPU/GPU·cache 절차를 쓰며, 과거와 조건을 확인하지 못하면 과거 시간은 참고값입니다. [시간 읽기와 주의점](NEGATIVE_PRUNING_RUNBOOK.ko.md#시간과-순위-확인)에서 파일명과 명령을 확인하세요.

<details>
<summary>내부 시간과 외부 시간을 구분하기</summary>

| 값 | 측정 범위 |
|---|---|
| `embedding_time_s` | 임베딩 모델·tokenizer 준비, 조각, 벡터·유사도·max 집계. 처음 데이터 읽기와 마지막 저장은 제외됩니다. |
| `invocation_time_s` | Score·데이터 읽기, 검증·정렬·gate, LM 구성·평가·논문별 결과 구성. 최종 요약·JSON 저장은 제외됩니다. |
| `estimated_embedding_plus_run_time_s` | 위 별도 구간의 추정 합. 전체 명령 wall time이 아닙니다. 재개·replay는 `null`입니다. |
| 기존 `time_s` | Full은 버전별 join 시간 또는 청크 합, early는 현재 invocation의 실행기 시간입니다. |

병렬 논문별 시간·shard 시간의 합, 재개 잔여 시간, 완료 재호출의 짧은 시간을 전체 시간으로 사용하지 않습니다. `manifest.json`이 이미 있으면 완료 논문이 0개여도 `resumed=true`입니다. 손실된 미완료 논문·chunk 호출과 HTTP 재시도도 완료 기록의 논리적 호출 합계에는 없습니다.

</details>

토큰 수·HTTP 재시도 수·parser 실패율·RP@5/10은 공통 출력이 아닙니다. 인접 upstream JoinAndRerank의 RP@K와 이 semi-join 지표를 섞지 않습니다. 실제 GPU 시간·모델 효과·품질은 서버 결과로 확인해야 하며 자동 테스트가 이를 대신하지는 않습니다.
