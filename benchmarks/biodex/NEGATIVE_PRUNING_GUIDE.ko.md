# Abstract proxy 실험 3·4·5 코드 안내

세 실험은 **동일한 abstract 조각과 기존 LLM 판정 코드**를 사용한다. 3번과 4번은 임베딩 모델을 비교하고, 4번과 5번은 논문 전체를 생략하는 negative pruning의 효과를 비교한다.

| 실험 | Proxy 모델 | LLM 평가 순서 | 논문 전체 생략 |
|---|---|---|---|
| 3 | E5-base-v2 | 논문–라벨 점수 내림차순 | 없음 |
| 4 | BioLORD-2023 | 논문–라벨 점수 내림차순 | 없음 |
| 5 | BioLORD-2023 | 4번과 동일 | 최대 점수가 threshold 미만이면 생략 |

논문–라벨 점수는 **abstract의 각 조각과 해당 라벨 사이 cosine similarity 중 최댓값**이다. LLM 입력은 세 실험 모두 기존 `fulltext_processed[:8000]`을 유지한다. Abstract가 그 8,000자에 전부 들어가는지는 이 구현이 가정하거나 보장하지 않는다.

## 1. 실행 순서부터 보기

| 순서 | 작업 | 필요한 입력 | LLM 호출 |
|---|---|---|---|
| 준비 1 | BioLORD validation 점수 생성 → threshold 결정 | Validation abstract + 정답 반응 | 0회 |
| 준비 2 | Test E5·BioLORD 점수 생성 | 같은 test 논문·라벨 | 0회 |
| 본 실험 | 3번·4번 실행 | 해당 모델의 test 점수 | 모든 논문, 첫 매칭에서 종료 |
| 빠른 비교 | 4번 결과에 threshold 두 개를 replay | 완료된 4번 + validation threshold | 추가 0회 |
| 시간 측정 | 선택한 threshold로 5번 실제 실행 | 같은 BioLORD test 점수 | 유지한 논문만 평가 |

**Threshold를 정하려고 4번 LLM 실험을 먼저 돌릴 필요는 없다.** Validation 점수와 정답만으로 결정한다. 4번을 먼저 확보하는 이유는 두 threshold의 F1·호출 수를 추가 LLM 실행 없이 비교하기 위해서다. Test 결과를 보고 threshold를 다시 조정하지 않고, validation에서 정한 두 정책을 그대로 보고한다.

4번 결과가 없어도 5번을 바로 실행할 수 있다. 보수적인 5번의 완료 결과에서 더 공격적인 threshold를 replay하는 것도 가능하다. 다만 원본에서 생략한 논문을 다시 살리는 조건은 해당 논문의 LLM 결과가 없어 재생할 수 없다.

## 2. 파일과 호출 경로

| 파일 | 읽을 내용 |
|---|---|
| [semijoin_pruning.py](semijoin_pruning.py) | Abstract 조각, 점수, threshold, 실험 3·4·5, replay와 지표 |
| [semijoin_baseline.py](semijoin_baseline.py) | 데이터 로더, 공통 LLM 설정, 기존 조기 종료 실행기 |
| [sentence_transformers_rm.py](../../lotus/models/sentence_transformers_rm.py) | 기존 LOTUS 임베딩 래퍼 |
| [sem_join.py](../../lotus/sem_ops/sem_join.py) | 논문–라벨 pair의 LOTUS 판정 경로 |
| [sem_filter.py](../../lotus/sem_ops/sem_filter.py) | 프롬프트 생성과 LLM 호출 연결 |
| [postprocessors.py](../../lotus/sem_ops/postprocessors.py) | LLM 응답을 Boolean으로 파싱 |
| [test_semijoin_pruning.py](../../tests/test_semijoin_pruning.py) | 길이·점수·gate·재개·replay·실제 LOTUS 연결 검증 |

```text
score      main → make_scores → load_articles(include_abstract=True)
           → 공통 abstract 조각 생성 → SentenceTransformersRM
           → 조각–라벨 유사도 → 논문별 max → 점수 JSON
calibrate  main → calibrate_file → calibrate → threshold JSON
run        main → run → output_lock → run_locked
           → gate + 라벨 순서 → 기존 run_early_stop
           → apply_gate → summarize → 결과 JSON
replay     main → replay → 원본 실행 검증 → apply_gate → summarize
```

LOTUS semantic operator 자체는 수정하지 않는다. 기존 실행기에 논문별 라벨 순서를 전달하고, 5번에서는 실행기에 넘기기 전에 논문을 거른다. 기존 1·2번 호출은 기본 동작을 유지한다.

기존 파일의 변경은 `lm_config()`로 동일 설정 재사용, `configure(with_retriever=False)`로 불필요한 E5 로드 생략, `load_articles()`의 split·abstract 선택, `run_early_stop()`의 선택적 `reaction_orders` 인자다. 기본값을 사용하는 기존 실행은 원래 입력과 라벨 순서를 유지한다.

## 3. 데이터 입력: proxy와 LLM을 구분하기

데이터 로더는 BioDEX의 지정 split 앞 N편을 읽고 원래 행 번호를 `_qid`로 유지한다.

```python
df["patient_description"] = df["fulltext_processed"].apply(
    lambda x: x[:TRUNCATION_LIMIT]
)
```

`TRUNCATION_LIMIT = 8000`이며 단위는 문자다. 이 값은 LLM의 논문 입력이다. Proxy 점수 생성에서는 별도로 가져온 `abstract`를 사용한다.

| 정보 | 쓰임 |
|---|---|
| `abstract` 전체 | E5·BioLORD proxy 임베딩 |
| `fulltext_processed[:8000]` | 기존 LLM pair 판정 |
| `reactions_list` | Validation threshold 보정과 최종 정답 평가 |

`REACTION_SET=liver` 또는 `haematologic`에서 표적 라벨 10개를 선택한다. 정답 반응 중 표적 라벨이 하나라도 있으면 article-level 정답은 양성이다.

```python
gold_matches = [
    index for index, label in enumerate(reactions)
    if label in row["reactions_list"]
]
```

정답은 임베딩 순서나 LLM 프롬프트에 넣지 않는다. LLM에는 논문 입력과 현재 반응 라벨만 전달한다. `input_sha256`는 abstract·LLM 입력·정답·ID를 함께 묶어 다른 데이터의 점수나 진행 기록을 섞지 못하게 한다.

## 4. Abstract를 어떻게 조각내는가

BioLORD의 기본 입력 길이는 128토큰이고 E5-base-v2는 512토큰이다. 세 실험은 비교 조건을 맞추기 위해 **양쪽 tokenizer에서 모두 128토큰 이내인 동일한 원문 조각**을 사용한다. 모델마다 토큰 경계가 다르므로 문자 수만으로 길이를 정하지 않는다.

```text
Abstract 전체
  ├─ 조각 1 ─┐
  ├─ 조각 2  ├─ 각각 임베딩 → 각 라벨과 유사도 계산
  └─ 조각 3 ─┘               → 라벨별 가장 큰 값 선택
```

각 후보 조각은 special token과 E5의 `passage: ` 접두어까지 포함해 실제 tokenizer 길이를 검사한다. 조각끼리 겹치지 않으며 조각을 순서대로 이어 붙이면 원문 abstract를 복원할 수 있다. 따라서 단순히 앞 128토큰만 쓰는 방식과 달리 뒷부분도 임베딩한다.

`fits_chunk()`가 양쪽 길이를 검사하고, `abstract_chunks()`가 검사에 통과한 원문 범위를 차례로 잘라 낸다. 가능하면 단어 사이에서 자르며, 그 경계도 다시 검사한다.

```python
chunks = [abstract_chunks(text, tokenizers) for text in articles["abstract"]]
flat_chunks = [text for group in chunks for text in group]
```

첫 줄은 논문별 조각 목록, 둘째 줄은 한 번에 임베딩할 전체 조각 목록이다. 원문 조각은 보존하고 임베딩 입력의 가장자리 공백만 제거한다.

| 임베딩 대상 | E5 입력 | BioLORD 입력 |
|---|---|---|
| Abstract 조각 | `passage: <조각>` | `<조각>` |
| 반응 라벨 | `query: <라벨>` | `<라벨>` |

128토큰은 이번 비교를 위한 공통 budget이다. E5의 512토큰 한도를 모두 사용하는 실험은 이번 3번 조건에 포함하지 않는다. 두 모델의 tokenizer는 길이 확인에 필요하지만 임베딩 가중치는 선택한 모델만 로드한다.

빈 abstract나 누락 abstract는 proxy를 사용할 수 없는 논문으로 처리한다. 본문으로 대체하지 않고 원래 라벨 순서로 LLM 평가하며 pruning하지 않는다.

## 5. 임베딩과 논문–라벨 점수

기존 LOTUS `SentenceTransformersRM`을 정규화 옵션으로 사용한다.

```python
rm = SentenceTransformersRM(
    model=MODELS[model_name], device=args.device, normalize_embeddings=True
)
```

조각이 C개, 라벨이 K개, 벡터 차원이 D라면 조각 벡터는 C×D, 라벨 벡터는 K×D다. 정규화 벡터의 내적은 cosine similarity다.

```text
한 논문에 대한 예시
                 라벨 A   라벨 B
조각 1            0.20     0.60
조각 2            0.80     0.30
논문의 최종 점수   0.80     0.60   ← 라벨별 max
```

각 조각과 라벨은 배치 임베딩 경로로 계산하고 모든 pair마다 모델을 다시 실행하지 않는다. 임베딩 후 행렬곱으로 유사도를 계산하고 논문별로 모은다.

```python
chunk_scores = vectors @ labels.T
# offset:stop은 현재 논문 조각들이 차지하는 범위
similarities[index] = chunk_scores[offset:stop].max(axis=0)
```

`axis=0`은 같은 라벨의 조각 점수 중 최대를 선택한다는 뜻이다. 실제 코드는 이 대입 전에 모든 조각의 벡터가 유효한지 검사한다.

NaN·무한대·영벡터 등 무효 임베딩이 있는 논문은 점수를 신뢰하지 않고 유지한다. 일부 조각만 정상이어도 나머지 조각에 중요한 정보가 있었을 수 있으므로 해당 논문의 proxy를 사용하지 않는다. 점수 JSON의 `null`은 사용 불가 점수를 뜻한다.

점수 파일의 `scorer`에는 모델·revision·입력 길이·정규화·abstract 조각 정책·두 tokenizer·접두어 정책을 기록한다. `scorer.chunking.tokenizers`의 SHA-256은 각 tokenizer의 실제 어휘·정규화·special token 처리 상태를 식별하며, 배치마다 달라질 수 있는 padding·truncation 상태는 제외한다. `chunks_sha256`은 논문별 원문 조각 목록을 식별한다. 기존 본문 prefix 점수와 threshold는 재사용하지 말고 다시 생성한다. 조각 수가 많은 논문은 우연히 큰 max를 얻을 가능성도 있으므로 threshold는 고정한 조각 정책과 라벨 패널에 한정해 해석한다.

## 6. 정렬과 negative pruning

점수 `[0.2, 0.8, 0.5]`의 라벨 순서는 `[1, 2, 0]`이다. 동점은 원래 순서를 유지하고 정상 점수에서는 모든 후보를 순서 목록에 남긴다.

```python
return sorted(range(len(scores)), key=lambda index: -scores[index])
```

3·4번은 이 순서로 모든 논문을 실행한다. 5번은 논문의 최대 라벨 점수가 threshold보다 작으면 그 논문 전체를 생략한다.

| 논문 | 라벨별 점수 | 최대 | Threshold 0.5에서 5번 동작 |
|---|---|---:|---|
| A | 0.70, 0.60, 0.30 | 0.70 | 0.70 라벨부터 LLM 평가 |
| B | 0.40, 0.25, 0.20 | 0.40 | 모든 라벨 평가 생략, 호출 0 |
| C | 0.50, 0.20, 0.10 | 0.50 | 평가함: threshold와 같으면 유지 |
| D | 무효 점수 포함 | 사용 불가 | 원래 라벨 순서로 평가 |

4·5번에는 **같은 BioLORD test 점수 파일**을 전달한다. 그러면 5번에서 유지한 논문의 순서·LLM 입력·판정 경로가 4번과 같다.

## 7. Threshold 보정: LLM 실행 없이 두 값만

`calibrate_file()`은 validation 점수에만 동작한다. 정답 양성 논문의 최대 점수를 모아 두 정책을 저장한다.

| 정책 | Validation에서 유지하려는 정답 양성 |
|---|---|
| `conservative` | 전부 |
| `retain_95` | 최소 95% |

허용 누락 수를 정수로 계산해 그 경계의 실제 점수를 threshold로 사용한다.

```python
allowed_misses = len(positives) - math.ceil(target * len(positives))
threshold = finite_positive_scores[
    min(allowed_misses, len(finite_positive_scores) - 1)
]
```

양성이 10편이면 95% 정책도 한 편을 버릴 수 없고 20편이면 최대 한 편을 버릴 수 있다. 동점 때문에 실제 누락은 더 적을 수 있으며 두 threshold가 같아져도 정상이다. 무효 점수의 논문은 항상 유지한다. 유효 점수 양성이 하나도 없으면 보정을 중단한다.

Calibration 파일과 test 점수의 모델·조각 정책·라벨 패널이 일치해야 한다. **Validation에서 100%를 남겼다고 test에서 false negative가 없다는 보장은 없다.** Test에서는 버린 양성 수와 최종 recall·F1을 함께 확인한다.

`run --experiment 3/4`는 threshold를 `None`으로 두어 gate를 끄고, `--experiment 5`는 유한한 threshold 또는 calibration 파일을 요구한다. 모델과 실험 번호가 다르거나 3·4번에 threshold를 지정하면 실행 전에 오류를 낸다.

## 8. 기존 LLM 실행기와 조기 종료

`run_locked()`는 점수로 유지할 논문과 라벨 순서를 정한 뒤 기존 실행기를 호출한다.

```python
baseline.run_early_stop(
    articles[articles["_qid"].isin(kept)],
    scores["reactions"], lm, str(progress_path),
    reaction_orders=orders,
)
```

`run_early_stop()`의 핵심 반복을 줄이면 다음과 같다.

```python
for k in order:
    out = pairwise_join(...)
    calls += 1
    if out.join_results:
        matched = reaction_list[k]
        break
```

첫 `True`에서 논문을 semi-join 결과에 포함하고 남은 라벨을 생략한다. 모든 라벨이 `False`이면 논문을 제외한다. 논문끼리는 `EARLY_WORKERS`개까지 병렬로 처리하고 한 논문의 라벨은 순차 평가한다.

```text
run_early_stop
  pairwise_join = lotus.sem_ops.sem_join.sem_join
    sem_filter
      filter_formatter                 프롬프트 생성
      CountingLM → LM → LiteLLM        LLM 요청
      filter_postprocess               True/False 판정
    join_results                       수락한 pair
```

공통 LOTUS parser는 응답을 해석하지 못하면 기본값 `True`를 사용한다. 따라서 첫 LLM witness는 **현재 parser가 수락한 첫 pair**다. 이 동작은 기존 비교군과 맞춰 유지한다.

## 9. 저장 파일과 replay

| 파일 | 내용 |
|---|---|
| 점수 JSON | 모델·조각 정책, 데이터 해시, 라벨, 논문별 점수·정답 |
| `manifest.json` | 실험 번호, 점수 해시, threshold와 출처, LLM·프롬프트·병렬도 |
| `progress.jsonl` | 평가 완료 논문별 `_qid`, `matched`, `lm_calls`, `time_s` |
| `articles.json` | 생략 논문까지 포함한 모든 논문의 결과·정답·순위 |
| `summary.json` | 전체 논문 기준 품질·호출 수·시간 |

`apply_gate()`는 생략된 논문에도 `pruned=true`, `matched=null`, `lm_calls=0`인 행을 만든다. 정답 양성이 생략되면 최종 F1의 FN에 포함된다. 모든 논문을 생략하면 LLM을 구성하지 않고 결과를 생성한다.

`replay`는 완료된 원본의 점수 해시·threshold·진행 기록·결과·요약을 대조한 다음 새 gate를 적용한다. 4번의 완료 결과에는 모든 논문의 LLM 결과가 있으므로 두 threshold를 추가 호출 없이 재생할 수 있다. 재생 결과는 호출 수와 품질을 비교할 수 있지만 **그 조건의 실제 실행시간은 제공하지 않는다**.

같은 폴더는 동일 설정 재개에만 사용한다. 출력 폴더는 실행기가 만들게 두고 로그와 외부 타이머 파일은 바깥에 저장한다. 마지막 JSONL을 쓰다 종료됐다면 미완성 꼬리만 복구하며 완료된 논문은 다시 평가하지 않는다.

## 10. 지표 수집 범위

| 팀 지표 | 저장 값 | 해석 |
|---|---|---|
| LLM calls | `summary.lm_calls`, 논문별 `lm_calls` | 완료된 논리적 pair 평가 수. HTTP 재시도 횟수와 다름 |
| First-witness rank | `mean_first_llm_witness_rank` | 선택된 논문에서 첫 LLM 수락 순위 평균. 오탐 포함 |
| 정답 witness 순위 | 논문별 `first_gold_witness_rank` | Proxy 순서에서 첫 정답 라벨 위치. 정답 양성 논문 기준으로 비교 |
| Embedding overhead | `embedding_time_s` | 모델·tokenizer 준비, 조각 생성, 임베딩, 유사도·max 집계 구간 |
| 전체 시간 | 외부 `/usr/bin/time` | 각 명령의 데이터 로드·저장까지 포함한 wall time |
| Article-level F1 | `f1`, `precision`, `recall`, `tp`, `fp`, `fn` | 생략된 논문을 포함한 원래 전체 test 집합 기준 |

`estimated_embedding_plus_run_time_s`는 별도로 잰 임베딩 구간과 현재 run 구간을 더한 추정치이며 전체 파이프라인 실측과 다르다. 재개 실행에는 전체 시간을 복원할 수 없어 이 값을 `null`로 기록한다. 병렬 논문별 `time_s`의 합은 wall time이 아니다.

이전에 `manifest.json`을 만든 실행에 다시 진입하면, 완료된 논문 기록이 아직 0개여도 `resumed=true`다. 이미 완료된 실행을 재호출할 때는 원래 요약과 최초 시간 정보를 보존한다.

`n_pruned`는 생략한 논문 수, `n_positive_pruned`는 그중 정답 양성 수, `positive_retention`은 gate가 남긴 정답 양성 비율이다. `n_positive_pruned`에는 4번의 LLM도 놓쳤을 논문이 포함될 수 있다. **4번 대비 추가 누락**은 4번이 선택한 정답 양성 ID 중 5번에서 사라진 ID를 비교해야 한다.

순위 평균은 대상 논문이 달라지면 단순 비교하기 어렵다. 3·4번의 정렬 품질은 같은 정답 양성 논문에 대한 `first_gold_witness_rank`로, 전체 비용은 같은 test 집합에 대한 호출 수로 비교한다.

토큰 수·HTTP 재시도 수·parser 실패율·RP@5·RP@10은 별도 출력하지 않는다. 중단된 미완료 논문에서 이미 발생한 요청도 완료 기록의 합계에는 없다. 비용·시간 본 실험은 새 폴더에서 중단 없이 실행한다.

## 11. 서버 명령: 복사해서 실행하기

Linux Bash 기준이며 저장소 루트에서 시작한다. LLM 추론 서버는 별도로 먼저 실행한다. 아래 모델명·프롬프트·병렬도는 기존 기본값 예시이므로 팀의 실제 비교군 설정과 맞춘다.

```bash
git fetch origin
git switch feat/biolord-negative-pruning
git pull --ff-only
python -m pip install -e . datasets
cd benchmarks/biodex

export REACTION_SET=liver
export PROMPT=paper
export LLM_MODEL=hosted_vllm/llama3.1-8b
export LLM_API_BASE=http://127.0.0.1:8080/v1
export LLM_BATCH=4
export EARLY_WORKERS=4

R=pruning_results/liver_paper
mkdir -p "$R"
```

`PROMPT=adr`는 다른 판정 질문이므로 비교군 전체에서 통일한다. 혈액 패널은 `REACTION_SET=haematologic`과 별도 `R` 경로로 전 과정을 반복한다. `--device cuda`는 proxy 임베딩 장치이며 CPU에서는 `cpu`로 바꾼다.

### 11.1 Validation 보정: LLM 0회

```bash
/usr/bin/time -f '%e' -o "$R/validation_biolord_wall_s.txt" \
  python semijoin_pruning.py score --model biolord --split validation --device cuda \
  --output "$R/validation_biolord.json" > "$R/validation_biolord.log" 2>&1

python semijoin_pruning.py calibrate "$R/validation_biolord.json" \
  --output "$R/thresholds.json"
```

Validation 전체를 사용한다. Calibration 파일의 두 정책과 관측 보존율을 먼저 확인한다.

### 11.2 같은 test 논문의 두 proxy 점수

아래는 기존 비교군과 맞춘 앞 250편 예시다. 전체 test를 사용하려면 **두 명령 모두** `--n-articles 250`을 생략한다.

```bash
/usr/bin/time -f '%e' -o "$R/test_e5_score_wall_s.txt" \
  python semijoin_pruning.py score --model e5 --split test --n-articles 250 --device cuda \
  --output "$R/test_e5.json" > "$R/test_e5_score.log" 2>&1

/usr/bin/time -f '%e' -o "$R/test_biolord_score_wall_s.txt" \
  python semijoin_pruning.py score --model biolord --split test --n-articles 250 --device cuda \
  --output "$R/test_biolord.json" > "$R/test_biolord_score.log" 2>&1
```

두 점수 파일의 `input_sha256`, `chunks_sha256`, `scorer.chunking`, `reactions`가 같아야 한다. 모델은 달라도 원본 논문·조각·tokenizer 정책·라벨은 동일하다는 확인이다. 전체 `scorer`의 모델과 접두어는 두 모델에서 다르므로 전체 객체가 같을 필요는 없다.

### 11.3 실험 3·4: pruning 없이 실행

```bash
/usr/bin/time -f '%e' -o "$R/experiment3_wall_s.txt" \
  python semijoin_pruning.py run "$R/test_e5.json" --experiment 3 \
  --output-dir "$R/experiment3" > "$R/experiment3.log" 2>&1

/usr/bin/time -f '%e' -o "$R/experiment4_wall_s.txt" \
  python semijoin_pruning.py run "$R/test_biolord.json" --experiment 4 \
  --output-dir "$R/experiment4" > "$R/experiment4.log" 2>&1
```

3·4번에는 threshold나 calibration을 전달하지 않는다. 3번은 E5 점수, 4번은 BioLORD 점수여야 한다.

### 11.4 실험 5의 두 정책: 우선 replay로 비교

완료된 4번을 재사용한다. 두 명령 모두 추가 LLM 호출이 없다.

```bash
python semijoin_pruning.py replay "$R/test_biolord.json" \
  --calibration "$R/thresholds.json" --policy conservative \
  --source-dir "$R/experiment4" --output "$R/experiment5_conservative_replay.json"

python semijoin_pruning.py replay "$R/test_biolord.json" \
  --calibration "$R/thresholds.json" --policy retain_95 \
  --source-dir "$R/experiment4" --output "$R/experiment5_retain95_replay.json"
```

### 11.5 선택한 5번 정책: 실제 시간 한 번 측정

보수적 정책을 주 결과로 삼는 예시다. 95% 정책을 실측하려면 정책과 출력 폴더·시간 파일 이름을 함께 바꾼다.

```bash
/usr/bin/time -f '%e' -o "$R/experiment5_conservative_wall_s.txt" \
  python semijoin_pruning.py run "$R/test_biolord.json" --experiment 5 \
  --calibration "$R/thresholds.json" --policy conservative \
  --output-dir "$R/experiment5_conservative" > "$R/experiment5_conservative.log" 2>&1
```

동일 폴더 재호출은 저장 결과를 반환하거나 미완료 논문만 재개한다. 이 짧은 재호출로 최초 시간 파일을 덮어쓰지 않는다. 다시 시간을 잴 때는 새 출력·시간 파일을 사용한다.

## 12. 결과 읽기

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path('pruning_results/liver_paper')
for name, score_time in [
    ('experiment3', 'test_e5_score_wall_s.txt'),
    ('experiment4', 'test_biolord_score_wall_s.txt'),
    ('experiment5_conservative', 'test_biolord_score_wall_s.txt'),
]:
    summary = json.loads((root / name / 'summary.json').read_text(encoding='utf-8'))
    rows = json.loads((root / name / 'articles.json').read_text(encoding='utf-8'))
    ranks = [r['first_gold_witness_rank'] for r in rows if r['gold_positive']]
    score_wall = float((root / score_time).read_text().strip())
    run_wall = float((root / f'{name}_wall_s.txt').read_text().strip())
    print(name, 'calls', summary['lm_calls'], 'F1', summary['f1'])
    print('mean first gold witness rank', sum(ranks) / len(ranks) if ranks else None)
    print('test score + run wall seconds', score_wall + run_wall)
PY
```

합계는 test 점수 생성과 live run을 별도로 실측한 시간의 합이며 단일 프로세스의 end-to-end 측정은 아니다. Validation 준비 시간은 별도로 기록한다. 4·5번이 점수 파일을 공유하더라도 각 방법의 전체 비용을 비교할 때는 각각 BioLORD 점수 생성 비용을 포함한다. 모델 다운로드·캐시·장치·병렬도 조건도 맞춘다.

## 13. 자동 검증과 서버 실험의 구분

자동 테스트는 공통 조각의 길이와 전체 abstract 보존, E5 prefix, max 집계, 무효 점수의 안전한 처리, threshold 경계, 원래 ID·평가 순서, 재개·replay 정합성, 기존 LOTUS pair 판정 경로를 확인한다. 실제 BioDEX 품질·GPU 실행시간·모델별 효과는 서버 실험에서 측정한다.
