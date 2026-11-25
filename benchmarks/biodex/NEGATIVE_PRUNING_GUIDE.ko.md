# Abstract proxy 실험 3·4·5 코드 안내

세 실험은 **동일한 abstract 조각과 기존 LLM 판정 코드**를 사용한다. 3번과 4번은 임베딩 모델을 비교하고, 4번과 5번은 논문 전체를 생략하는 negative pruning의 효과를 비교한다.

| 실험 | Proxy 모델 | LLM 평가 순서 | 논문 전체 생략 |
|---|---|---|---|
| 3 | E5-base-v2 | 논문–라벨 점수 내림차순 | 없음 |
| 4 | BioLORD-2023 | 논문–라벨 점수 내림차순 | 없음 |
| 5 | BioLORD-2023 | 4번과 동일 | 최대 점수가 threshold 미만이면 생략 |

논문–라벨 점수는 **abstract의 각 조각과 해당 라벨 사이 cosine similarity 중 최댓값**이다. LLM 입력은 세 실험 모두 기존 `fulltext_processed[:8000]`을 유지한다. Abstract가 그 8,000자에 전부 들어가는지는 이 구현이 가정하거나 보장하지 않는다.

### 서버에서 남은 작업

**70B GPU에서 3 → 4 → 5만 새로 실행한다.** 1·2번은 같은 70B·paper·test 앞 250편 조건의 서버 기록을 인용한다. 원본 요약과 선택 논문 CSV가 남아 있지 않아 이 두 행의 호출 수·라벨 F1·외부 경과시간은 별도 표로 보고한다.

| 실험 | 논리적 호출 수 | 라벨 기준 F1 | Python 명령 외부 경과시간 |
|---|---:|---:|---:|
| 1. 전체 조인 | 2,500 | 0.172 | 7,282초 |
| 2. 고정 순서 조기 종료 | 1,855 | 0.172 | 1,896초 |

3~5번의 선택 논문과 정답을 새로 집계한다. 과거 전체 조인과의 기사별 일치도는 전체 선택 목록이 있어야 계산할 수 있다. 기록의 출처·실행 조건·복원이 어려운 항목은 [과거 70B 실험 결과](HISTORICAL_70B_RESULTS.ko.md)에 정리했다.

## 1. 실행 순서부터 보기

| 순서 | 작업 | 필요한 입력 | LLM 호출 |
|---|---|---|---|
| 공통 준비 | 70B GPU 서비스 확인·점수 계산용 파일 사전 준비 | 현재 배포 설정과 과거 실행 조건 | 사전 점검만 |
| **3번** | E5 test 점수 생성 → 과거 정답 양성 인덱스·패널 순서 대조 → 로컬 LLM 점검 → 3번 실행 | Test abstract + E5 점수 | 첫 매칭에서 종료 |
| 4번 준비 | BioLORD test·validation 점수 생성 → threshold 고정 | Validation abstract + 정답 반응 | 0회 |
| **4번** | 두 proxy의 입력 일치 확인 → 로컬 LLM 점검 → 4번 실행 | 3번과 같은 test 논문·라벨 | 첫 매칭에서 종료 |
| **5번** | 완료된 4번으로 두 정책 replay → 보수적 5번 실측 | 같은 BioLORD 점수 + 고정 threshold | Replay 0회, 실측은 유지한 논문만 |

**Threshold를 정하려고 4번 LLM 실험을 먼저 돌릴 필요는 없다.** Validation 점수와 정답만으로 결정한다. 4번을 먼저 확보하는 이유는 두 threshold의 F1·호출 수를 추가 LLM 실행 없이 비교하기 위해서다. Test 결과를 보고 threshold를 다시 조정하지 않고, validation에서 정한 두 정책을 그대로 보고한다.

이번 실행 순서는 **3 → 4 → 5**로 고정한다. 5번의 주 조건은 `conservative`, 보조 조건은 `retain_95`이며 test 결과를 보기 전에 정한다. 두 정책이 같은 threshold이면 동일 조건으로 보고한다. 이미 저장한 점수는 재생성하지 않고 재사용한다.

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
check-baselines   test 점수 + 현재 설정 + 기존 요약 → 기록된 항목 대조 보고서
run        main → run → output_lock → run_locked
           → 현재 입력 확인 → 선택적 과거 요약 대조 → gate + 라벨 순서
           → 기존 run_early_stop
           → apply_gate → summarize → 결과 JSON
replay     main → replay → 원본 실행 검증 → apply_gate → summarize
report     main → report → completed_run + 과거 요약/CSV 검증
           → comparison_prf → 공통 비교 JSON (LLM 호출 없음)
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

요약의 `manifest_sha256`으로 LLM·프롬프트·실험 설정과 결과를 연결해 다른 실행의 manifest가 섞인 경우도 거부한다. 이 필드가 없는 이전 결과는 최신 형식으로 새 실행을 준비해야 한다. 점수 생성은 기존 출력 파일 덮어쓰기를 거부한다. 점수 내용이 같아도 새로 잰 임베딩 시간이 달라지면 파일 해시가 달라지므로, 재개·replay에는 최초 점수 파일을 그대로 보존한다.

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

## 11. 서버 명령: 3 → 4 → 5 순서대로

Linux Bash 기준이다. **실제 LLM 평가 순서는 3번 run → 4번 run → 5번 run**이다. 점수 생성·validation 보정·replay는 LLM을 호출하지 않는다. 각 단계가 성공하고 저장 파일을 확인한 뒤 다음 단계로 진행한다.

### 11.0 공통 준비: 70B GPU 서비스와 실행 환경

먼저 사용 가능한 GPU 노드와 팀의 70B 서비스 배포를 확인한다. 과거 CloudLab 예약의 현재 접속 가능 여부는 확인되지 않았다. 서비스가 준비되어 있으면 기존 endpoint를 사용한다. 중지된 서비스는 팀의 배포 절차로 준비한다. 아래 명령은 서버를 기동하지 않는다.

| 항목 | 이번 비교의 기준 |
|---|---|
| 모델·장비 | 과거 기록은 `bartowski/Meta-Llama-3.1-70B-Instruct-GGUF`, `Q4_K_M`, V100 16GB 4장. 현 서버의 가중치·장비·배포 정보를 확인하고 차이를 기록 |
| LLM 설정 | 같은 paper 질문, temperature 0, 출력 최대 256토큰, `LLM_BATCH=8`, `EARLY_WORKERS=8` |
| 서버 설정 | 과거 실행 인자는 `-ngl 99 -c 65536 --parallel 8 -fa on -ctk q8_0 -ctv q8_0 --no-jinja --chat-template llama3`. 현 서비스의 설정을 대조하는 참고 기록 |
| 논문·라벨 | BioDEX test 앞 250편, 본문 앞 8,000자, 같은 순서의 간 라벨 10개와 정답 양성 10편 |
| Proxy 장치 | `EMBED_DEVICE=cpu` 또는 `cuda`. 세 실험에서 같은 장치를 사용 |
| 저장 경로 | 새 비교 디렉터리 `R`. experiment3/4/5 폴더는 실행기가 생성 |

`hosted_vllm/`은 LiteLLM의 OpenAI 호환 서버 연결 이름이다. 서버 운영 설정에서 70B 가중치를 확인한 후 해당 서비스의 실제 모델 ID를 사용한다. 코드의 일반 기본값은 8B이므로 이 가이드에서는 모델을 명시한다. Alias만으로 같은 가중치를 입증할 수 없고, 과거의 dataset revision과 llama.cpp 빌드 커밋은 미확인이다.

#### A. Python 환경과 코드

서버에서 팀의 LOTUS Conda/venv 또는 환경 설정 스크립트를 먼저 활성화한 뒤 저장소 루트에서 시작한다. 기존 패키지 버전을 유지하기 위해 현재 checkout만 editable로 연결한다.

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

`cuda` 임베딩은 CUDA 지원 PyTorch와 여유 VRAM이 있을 때만 사용한다. LLM이 GPU에서 실행돼도 proxy는 CPU로 계산할 수 있다. 기존 LLM 서비스와 GPU 메모리가 겹치면 proxy를 CPU나 다른 GPU에서 계산한다. 기존 서비스의 중지·재시작이 필요한 경우에는 서버의 원래 운영 절차만 사용한다. 모델 다운로드를 포함한 cold 비용과 캐시가 준비된 비용도 구분한다.

새 환경을 CI와 같은 의존성 버전으로 맞추려면 저장소 루트에서 `python -m pip install uv`, `uv export --locked --no-emit-project --no-hashes --all-extras -o /tmp/lotus-constraints.txt`, `uv pip install --python "$(command -v python)" -e . datasets --constraint /tmp/lotus-constraints.txt`를 사용한다. 기존 환경을 사용할 때에는 설치된 버전을 기록한다.

#### B. 기존 endpoint·설정 유지와 기록

실제 70B GPU 배포의 모델 ID를 확인하고 `LLM_MODEL=hosted_vllm/<실제 ID>` 형식으로 내보낸 다음 실행한다. 클라이언트와 서버가 다른 호스트라면 `LLM_API_BASE`도 해당 서버 주소로 지정한다. 서버에 기존 파일 경로가 있으면 `MODEL_GGUF`, `LLAMA_SERVER`에서 파일 SHA를 추가로 기록할 수 있다.

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
export EMBED_DEVICE="${EMBED_DEVICE:-cpu}"
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

GPU 서버에서 `nvidia-smi -L` 출력을 이번 실행의 `$R/gpu.txt`에 보존하고, 클라이언트가 다른 호스트이면 서버에서 가져온 기록임을 표시한다. 기존 설정 파일·시작 로그에서 가중치 revision·양자화·서버 버전·시작 명령을 확인할 수 있으면 **위 fingerprint를 생성하기 전에** 그 내용을 `server-config.txt` 생성 블록에 추가한다. 인증 정보는 기록하지 않는다. Endpoint URL에도 비밀 키를 넣지 않는다. 기존 인증은 환경에서 재사용하며 `settings.sh`에 키를 저장하지 않는다.

`R`은 비교 전체에서 한 번만 만든다. 새 터미널에서는 같은 환경과 기존 인증을 활성화하고, 출력된 절대 경로의 `settings.sh`를 source한다. `LLM_SERVER_ID`는 기록한 설정의 식별자다. 파일 SHA를 확보하지 못했다면 **기록의 일치만 확인하며 실제 가중치 동일성을 증명하지 않는다.** 미확인 정보 때문에 새 모델·새 설정을 선택하지 않고 기존 서비스를 유지한다.

`EARLY_WORKERS`는 동시 논문 수, `LLM_BATCH`는 한 LM 호출 안의 상한, 서버 슬롯 수는 기존 서비스 설정이다. 세 값을 같은 의미로 해석하지 않으며 서버 context·슬롯·GPU offload를 이 PR의 예시로 덮어쓰지 않는다. 본문 8,000문자와 프롬프트·출력을 수용하는 context인지 기존 시작 로그에서 확인한다.

#### C. 기존 서버의 운영·캐시 절차 유지

이미 서비스가 실행 중이면 그 endpoint를 그대로 사용한다. 서비스가 중지돼 있으면 **서버에 원래 준비된 시작 스크립트·서비스·작업 절차로 복구**한다. 이 문서의 추정값으로 별도 `llama-server`를 띄우지 않는다.

각 live run 전에 같은 운영·캐시 절차를 적용하고 `SERVER_CACHE_POLICY`에 기록한다. 원래 재시작/캐시 초기화 절차를 이용할 수 있으면 3·4·5번에 동일하게 적용한다. 그대로 실행 중인 서비스를 쓸 경우에는 초기 KV cache 상태가 통제되지 않았음을 기록하고, 그 시간값을 cold-start 비교로 해석하지 않는다. 기존 1·2번과 시간 비교할 때도 측정 범위와 캐시 조건을 확인한다. LOTUS 응답 cache는 기본 False이며 서버 KV cache와 별개다.

#### D. 실험 터미널: 실제 LOTUS 로컬 연결 사전 점검

아래 함수를 한 번 정의한다. 각 run 전에 서버가 준비됐는지와 실제 LOTUS → LiteLLM → 로컬 서버 경로를 같은 짧은 요청으로 확인한다. 사전 점검 시간·호출은 본 실험 지표 밖에 둔다.

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

기본 로컬 경로는 외부 OpenAI API 키가 필요 없다. 기존 서버 인증이 필요하면 기존 `HOSTED_VLLM_API_KEY` 환경값을 사용한다. 위 curl 점검도 이를 재사용하며 키를 기록하지 않는다. `/health` 같은 엔진 전용 endpoint는 선택 점검이며 공통 절차는 `/models`와 실제 LOTUS 요청으로 확인한다. 접속 오류·alias 불일치·빈 응답이 있으면 본 실험을 시작하지 않는다. LOTUS 응답 cache는 기본 False이며 서버의 prompt/KV cache와 별개다.

#### E. 시간 측정 전 데이터·임베딩 모델 파일 준비

본 측정과 같은 사용자·환경·캐시 경로에서 한 번 실행한다. Test·validation 데이터, E5와 BioLORD 가중치·tokenizer의 최초 다운로드를 이 단계에서 마치고 준비 로그를 남긴다. 새 Python 프로세스가 파일을 메모리로 읽는 시간과 일반적인 Hub 메타데이터 조회 지연은 test score 명령 시간에 포함될 수 있다. 측정 로그에서 실제 파일 다운로드를 확인하면 해당 시간을 ‘네트워크·다운로드 포함’으로 표시한다.

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

### 11.1 실험 3: E5 점수 → 서버 점검 → 3번 실행

사전 설정: 위 공통 환경, 같은 논문 수·패널·프롬프트, 새 출력, proxy 계산용 메모리를 확인한다. Score는 LLM 없이 실행된다.

```bash
fresh "$R/test_e5.json" "$R/test_e5_score_wall_s.txt" "$R/test_e5_score.log"
/usr/bin/time -f '%e' -o "$R/test_e5_score_wall_s.txt" \
  python semijoin_pruning.py score --model e5 --split test --n-articles "$N_ARTICLES" \
  --device "$EMBED_DEVICE" --output "$R/test_e5.json" > "$R/test_e5_score.log" 2>&1
```

과거 기록에 있는 **test 행 인덱스와 라벨 순서**를 E5 점수 파일에서 대조한다. 행 번호는 0부터 센다. 같은 인덱스에서 양성 10편이 일치해도 과거 데이터 revision과 250편 전체의 원문 동일성은 미확인으로 남는다.

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

대조가 끝나면 3번을 실행한다. 3번에는 threshold를 넣지 않는다.

```bash
preflight experiment3
fresh "$R/experiment3" "$R/experiment3_wall_s.txt" "$R/experiment3.log"
/usr/bin/time -f '%e' -o "$R/experiment3_wall_s.txt" \
  python semijoin_pruning.py run "$R/test_e5.json" --experiment 3 \
  --output-dir "$R/experiment3" > "$R/experiment3.log" 2>&1
test -s "$R/experiment3/summary.json"
test -s "$R/experiment3/articles.json"
```

완료되면 `experiment3/manifest.json`, `progress.jsonl`, `articles.json`, `summary.json`이 생성된다. 기존 서버 운영 설정을 유지한 채 4번 점수를 준비한다. **저장된 score를 다시 생성하지 않는다.** 중단됐다면 기존 score·timer를 보존하고 11.4 재개 절차를 쓴다.

### 11.2 실험 4: BioLORD 점수·threshold 준비 → 4번 실행

사전 설정: 3번 완료, 같은 논문 수·패널·프롬프트·LLM 설정을 확인한다. Proxy 장치는 기존 LLM 서비스와 메모리가 충돌하지 않도록 선택한다. Validation 전체로 두 threshold를 정하고 **4번 test의 LLM 결과를 보기 전에 고정**한다. Threshold 계산 자체는 4번 LLM 실행이 필요 없다.

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
assert bio['scorer'] == validation['scorer'] == calibration['scorer']
assert bio['reactions'] == calibration['reactions']
assert len(e5['articles']) == len(bio['articles']) == int(os.environ['N_ARTICLES'])
print('Shared abstracts/chunks/panel and calibration OK')
print(json.dumps(calibration['policies'], indent=2))
PY
```

불일치하면 진행하지 않는다. Model·tokenizer 버전이 달라졌다면 새 비교 폴더에서 동일 조건으로 다시 준비한다. 기존 score를 덮어써서 맞추지 않는다. 기존 endpoint·모델·운영/캐시 절차를 유지한 상태에서 4번 연결 점검을 진행한다. **4번에도 threshold를 넣지 않는다.**

```bash
preflight experiment4
fresh "$R/experiment4" "$R/experiment4_wall_s.txt" "$R/experiment4.log"
/usr/bin/time -f '%e' -o "$R/experiment4_wall_s.txt" \
  python semijoin_pruning.py run "$R/test_biolord.json" --experiment 4 \
  --output-dir "$R/experiment4" > "$R/experiment4.log" 2>&1
test -s "$R/experiment4/summary.json"
test -s "$R/experiment4/articles.json"
```

저장 파일 구성은 3번과 같다. 4번 결과와 BioLORD score를 그대로 5번에 쓴다.

### 11.3 실험 5: 두 정책 replay → 보수적 조건 1회 실측

사전 설정: 4번 완료, validation threshold 고정, **4번과 같은 BioLORD test score**, 같은 LLM 설정을 확인한다. 먼저 저장된 4번 판정으로 두 정책을 replay한다. 서버가 정지 중이어도 되며 새 LLM 호출은 없다.

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

주 조건은 사전에 정한 `conservative`이며 test 결과를 보고 바꾸지 않는다. 같은 기존 서비스와 운영/캐시 절차를 사용해 실제 시간을 잰다.

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

전부 prune되면 `progress.jsonl`이 없어도 정상이다. `retain_95`를 replay만 했다면 실제 시간은 미측정으로 표시한다. 이 조건도 실측하려면 policy·폴더·timer·log 이름을 모두 분리하고 같은 기존 서버 운영/캐시 절차와 사전 점검을 적용한다.

### 11.4 중단 시 재개

같은 설정·score·출력 폴더로 실행하면 미완료 논문만 재개한다. **score 생성이나 원래 timer를 다시 실행하지 않는다.** 아래는 3번 예시다. 사전 점검도 새 이름으로 실행하고 재개 log·timer에 새 번호를 쓴다.

새 실험 터미널에서는 원래 Python 환경을 활성화하고 `source /실제/기존/R/settings.sh` 후 `cd "$BENCHMARK_DIR"`를 실행한다. 위 D단계는 `fresh`·`preflight` 함수도 settings 파일에 보존하므로 재접속 후 사용할 수 있다. 서버도 끊겼다면 기존 서버의 원래 서비스·시작 절차로 복구한다. 새 모델이나 추정 실행 인자로 대체하지 않는다.

```bash
preflight experiment3_resume1
fresh "$R/experiment3_resume1_wall_s.txt" "$R/experiment3_resume1.log"
/usr/bin/time -f '%e' -o "$R/experiment3_resume1_wall_s.txt" \
  python semijoin_pruning.py run "$R/test_e5.json" --experiment 3 \
  --output-dir "$R/experiment3" > "$R/experiment3_resume1.log" 2>&1
```

4번은 원래 점수와 번호를, 5번은 원래 calibration·policy까지 그대로 사용한다. 재개 시간은 남은 작업 시간이며 전체 시간이 아니다. 첫 미완료 논문에서 잃은 pair 호출도 있어 단순 합산을 무중단 실험과 같다고 보고하지 않는다. 완료 폴더 재호출은 저장 결과를 짧게 반환하므로 원래 timer를 덮어쓰지 않는다. 시간 본 결과가 필요하면 새 출력 폴더에서 무중단 실측한다. 새 코드의 manifest/summary 식별 필드가 없는 옛 실행을 수동 편집해 재사용하지 않는다. 유효한 기존 score는 보존하고 새 실행 폴더를 사용한다.

다른 패널과 전체 test는 별도 연구 조건이다. 현재 70B 과거 수치는 간 패널 250편에만 사용한다.

## 12. 결과 읽기와 설정 비교

위에서 사용한 같은 `R`을 읽는다. 재개 실행은 전체 시간을 표시하지 않고, 완료 재호출로 timer가 짧게 덮인 정황도 검사한다. `fresh` 사전 검사와 함께 사용한다.

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

합계는 test score 생성과 live run을 별도로 측정한 시간의 합이며 단일 프로세스 end-to-end 실측은 아니다. Validation 준비 시간은 별도다. 4·5번이 score를 공유하더라도 각 방법의 비용에는 각각 BioLORD score 생성 비용을 포함한다. 3·4번 gold 순위는 같은 정답 양성 ID 전체로 비교하고, 5번의 순서가 4번보다 좋아졌다는 근거로 쓰지 않는다.

과거 1번 7,282초와 2번 1,896초는 서버가 이미 기동된 상태에서 Python 명령 시작부터 종료까지 잰 외부 경과시간이다. 새 실험은 최초 파일 다운로드·서버 준비·사전 점검을 따로 수행한다. 본 측정 중 발생한 Hub 메타데이터 조회·추가 다운로드 시간은 외부 타이머에 포함된다. 과거와 GPU 종류·수, 가중치·서버 설정, 캐시 정책의 일치 여부를 기록한다. 확인되지 않은 차이가 있으면 시간은 출처·조건과 함께 나열하고 같은 환경에서 얻은 속도비라고 보고하지 않는다.

`server-config.txt`와 기존 서버의 설정·운영/캐시 절차를 확인한다. GGUF SHA·바이너리 버전·실제 시작 명령은 확인 가능한 경우 함께 보존하며, 미확인 항목은 동일성을 검증한 것처럼 쓰지 않는다. Live 결과 차이는 4번의 동일 판정을 쓰는 replay와 분리해 LLM 응답 변동도 검토한다.

### 12.1 실험 3~5의 품질 집계와 과거 1·2번 인용

다음 명령은 이번에 측정한 3~5번과 두 replay만 집계한다. 과거 1·2번의 수치는 [70B 실험 기록](HISTORICAL_70B_RESULTS.ko.md)에서 인용한다. 원본 선택 CSV와 요약이 없어 과거 F1이나 호출 수만으로는 `report`에 넣을 수 없다.

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

`report`는 저장된 점수와 결과만 읽는다. 데이터셋을 다시 내려받거나 LLM을 호출할 필요가 없다. 각 `--run`에는 그 실험에 사용한 점수 파일과 완료 폴더를 한 쌍으로 전달한다. 3·4·5번의 논문 순서·정답·라벨 목록·공통 조각과 저장된 LLM 설정을 대조한다. Replay를 포함할 때는 원본 live run도 전달한다. 위 두 replay의 원본은 4번이다.

| `comparison.json` 항목 | 읽는 방법 |
|---|---|
| `results[].vs_labels` | 해당 방법이 선택한 논문 ID 집합을 같은 test 정답 양성 논문 ID 집합과 비교 |
| `results[].vs_join_full` | 같은 선택 ID 집합을 1번 full join이 선택한 논문 ID 집합과 비교. 1번을 제공하지 않은 부분 보고서에서는 `null` |
| `n_articles`, `n_positive`, `n_selected` | 같은 test 전체 편수, 정답 양성 편수, 각 방법이 선택한 고유 논문 수 |
| `experiment`, `mode`, `source`, `lm_calls`, `lm_calls_kind` | 실험·원본·논리 호출 수. 같은 5번도 live와 replay를 별도 행으로 표시 |
| `threshold`, `threshold_source` | 적용한 경계와 기록된 출처. 보정한 두 정책의 경계가 같으면 `threshold_source.policy`로 정책명을 구분 |
| `reference_scores_sha256` | 정답 기준으로 읽은 `--scores` JSON 내용의 식별자 |
| `missing_experiments`, `baseline_comparison` | 제공된 결과가 없는 실험 번호와 과거 기록에서 대조한 항목·미확인 항목 |

이번 명령의 예상값은 `missing_experiments: [1, 2]`와 모든 행의 `vs_join_full: null`이다. `vs_labels`는 article-level TP·FP·FN, 선택 수와 소수점 네 자리로 반올림한 precision·recall·F1을 제공한다. `lm_calls`는 논리적 평가 횟수이고, replay에서는 4번 판정으로 재생한 정책의 비용이다. 시간과 witness 순위는 12절의 출력에서 별도로 읽는다.

원본이 나중에 복구되면 선택적으로 `--baseline-summary`를 추가해 집계할 수 있다. 각 요약과 같은 디렉터리에 1번은 `join_full_semi.csv` 또는 `join_full_join.csv`, 2번은 `early_stop_semi.csv`가 필요하다. 기존 요약의 선택 편수와 CSV가 맞아야 한다. 과거의 정답 양성 인덱스나 부분 진행 기록으로 선택 목록을 대체할 수 없다.

## 13. 자동 검증과 서버 실험

자동 테스트는 chunk 길이·전체 abstract 보존·E5 prefix·max 집계·무효 점수·threshold 경계·원래 ID·평가 순서·재개·replay·기존 LOTUS 판정 경로를 확인한다. 실제 BioDEX 품질·GPU 시간·모델 효과는 서버에서 측정한다. 이 테스트가 모델 다운로드나 실제 GPU 추론을 대신하지는 않는다.

공통 집계 테스트는 1~5번과 replay의 같은 출력 형식, 기준 없음과 실제 빈 기준의 구분, 원본 불변, 모델·데이터 추가 호출 없음, CSV·완료 기록·gold·참조 파일 불일치 거부, 4·5번 점수 파일 일치, 기존 P/R/F1 반올림 경계값을 확인한다.

로컬 경로와 설정의 근거는 [LiteLLM OpenAI 호환 vLLM 경로](https://docs.litellm.ai/docs/providers/vllm), [llama.cpp server 공식 설명](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)이다. 과거 배포의 설정과 확인되지 않은 항목은 [70B 실험 기록](HISTORICAL_70B_RESULTS.ko.md)에 정리했다.
