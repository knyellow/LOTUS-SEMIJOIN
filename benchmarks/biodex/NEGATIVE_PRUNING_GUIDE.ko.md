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
| 공통 준비 | 1·2번의 환경·로컬 GGUF·서버 설정 확인 | 같은 가중치·양자화·프롬프트·패널·논문 수 | 사전 점검만 |
| **3번** | E5 test 점수 생성 → 로컬 LLM 점검 → 3번 실행 | Test abstract + E5 점수 | 첫 매칭에서 종료 |
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

### 11.0 공통 준비: 기존 실험 서버의 세팅 재사용

**기본 절차는 1·2번을 실행한 서버에 이미 준비된 Python 환경·LLM 서비스·가중치·운영 설정을 그대로 이용하는 것이다.** 사용자가 지금 GGUF 파일명이나 시작 명령을 별도로 알아내야 진행할 수 있는 절차가 아니다. LLM 가중치의 새 다운로드·새 서버 설치·임의의 context/양자화/GPU 설정으로 재구성하는 작업은 필요하지 않다. E5·BioLORD proxy 모델은 별개이며 score 단계에서 캐시가 없으면 다운로드한다. 서버에 접속한 뒤 기존 환경을 활성화하고, 확인 가능한 설정을 기록한 후 3→4→5를 실행한다.

| 설정 | 확인할 내용 |
|---|---|
| Python | 기존 1·2번 환경 우선 재사용. 새 환경은 Python 3.10~3.12 |
| 논문·라벨·질문 | 기존 1·2번과 같은 N·panel·PROMPT. 예시는 앞 250편, liver, paper |
| LLM | 기존 1·2번의 같은 GGUF 파일·양자화·revision |
| 서버 | 같은 llama.cpp 바이너리·chat template·context·슬롯 수·GPU 설정 |
| Proxy 장치 | EMBED_DEVICE=cpu 또는 cuda. LLM 추론 장치와 별개 |
| 저장 경로 | 이번 비교 전체의 새 R. experiment3/4/5 폴더는 실행기가 생성 |

기본 경로는 `hosted_vllm/llama3.1-8b` → `http://localhost:8080/v1`이다. `hosted_vllm/`은 LiteLLM의 OpenAI 호환 서버 연결 이름이며 실제 서버가 반드시 vLLM이어야 한다는 뜻은 아니다. **이미 준비된 LLM 서비스에 연결한다.** 코드는 LLM을 새로 다운로드하거나 서버를 시작하지 않으며 외부 유료 OpenAI 모델로 자동 전환하지 않는다.

인접 `biodex_local.py`에는 llama.cpp·Q4_K_M이 명시돼 있고 초기 N=25 커밋에는 Llama-3.1-8B Q4가 적혀 있다. 다만 **완료된 N=250 실험 1·2번의 실제 GGUF·양자화·서버 명령은 이 기록만으로 확정할 수 없다.** 아래 alias는 코드 기본값 예시다. 기존 서버에 설정된 endpoint·alias·가중치를 우선하며 새 모델로 대체하지 않는다. 파일명·양자화·시작 명령은 서버의 기존 로그·설정에서 확인되는 범위만 기록하고 미확인 항목은 그대로 표시한다. Alias 일치는 가중치 동일성의 증명이 아니다.

#### A. Python 환경과 코드

서버에서 기존 1·2번용 Conda/venv 또는 팀의 환경 설정 스크립트를 먼저 활성화한 뒤 저장소 루트에서 시작한다. 기존 패키지 버전을 유지하기 위해 현재 checkout만 editable로 연결한다.

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

새 환경을 CI와 같은 의존성 버전으로 맞추려면 저장소 루트에서 `python -m pip install uv`, `uv export --locked --no-emit-project --no-hashes --all-extras -o /tmp/lotus-constraints.txt`, `uv pip install --python "$(command -v python)" -e . datasets --constraint /tmp/lotus-constraints.txt`를 사용한다. 기존 1·2번 환경을 유지할 때는 패키지를 임의로 업그레이드하지 않는다. 서버에 GGUF가 없다면 기존 배포처·revision·정확한 파일명을 확인해 같은 파일을 복원하고 SHA를 대조한다.

#### B. 기존 endpoint·설정 유지와 기록

기존 환경에 이미 있는 값은 보존하고, 비어 있는 값에만 코드 기본값을 넣는다. `LLM_MODEL`과 `LLM_API_BASE`는 기존 팀 환경 설정을 우선한다. 기본값과 실제 서버가 다르면 `/models` 확인 결과와 기존 설정에 맞춘다. `LLAMA_SERVER`·`MODEL_GGUF` 경로를 새로 채우는 것은 필수가 아니다. 로컬 연결에서는 `hosted_vllm/<기존 서버 모델 ID>` 형식만 허용해 `LLM_API_BASE`가 실제 LOTUS 요청에 적용되도록 검사한다. 다른 provider 접두어가 있으면 기존 서버의 모델 ID는 유지하고 연결용 접두어만 맞춘다.

```bash
export REACTION_SET="${REACTION_SET:-liver}"
export PROMPT="${PROMPT:-paper}"
export LLM_MODEL="${LLM_MODEL:-hosted_vllm/llama3.1-8b}"
export LLM_API_BASE="${LLM_API_BASE:-http://127.0.0.1:8080/v1}"
case "$LLM_MODEL" in
  hosted_vllm/?*) ;;
  *) printf '로컬 경로는 hosted_vllm/<기존 서버 모델 ID>를 사용하세요. 가중치는 변경하지 않습니다.\n' >&2; exit 1 ;;
esac
export LLM_BATCH="${LLM_BATCH:-4}"
export EARLY_WORKERS="${EARLY_WORKERS:-4}"
export EMBED_DEVICE="${EMBED_DEVICE:-cpu}"
export N_ARTICLES="${N_ARTICLES:-250}"
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
  printf 'source=existing_experiments_1_2_server\n'
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

기존 설정 파일·시작 로그에서 가중치 revision·양자화·서버 버전·시작 명령을 확인할 수 있으면 **위 fingerprint를 생성하기 전에** 그 내용을 `server-config.txt` 생성 블록에 추가한다. 인증 정보는 기록하지 않는다. Endpoint URL에도 비밀 키를 넣지 않는다. 기존 인증은 환경에서 재사용하며 `settings.sh`에 키를 저장하지 않는다.

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

### 11.1 실험 3: E5 점수 → 서버 점검 → 3번 실행

사전 설정: 위 공통 환경, 같은 논문 수·패널·프롬프트, 새 출력, proxy 계산용 메모리를 확인한다. Score는 LLM 없이 실행된다.

```bash
fresh "$R/test_e5.json" "$R/test_e5_score_wall_s.txt" "$R/test_e5_score.log"
/usr/bin/time -f '%e' -o "$R/test_e5_score_wall_s.txt" \
  python semijoin_pruning.py score --model e5 --split test --n-articles "$N_ARTICLES" \
  --device "$EMBED_DEVICE" --output "$R/test_e5.json" > "$R/test_e5_score.log" 2>&1
```

기존 LLM 서비스가 준비된 상태에서 다음 점검과 실행을 진행한다. **3번에는 threshold를 넣지 않는다.**

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

저장 파일 구성은 3번과 같다. 4번 결과와 BioLORD score를 그대로 5번에 쓴다. 전체 test를 쓰려면 두 test score 명령 모두 `--n-articles`를 생략하고 위 건수 assert도 실제 전체 건수로 맞춘다.

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

혈액 패널은 `REACTION_SET=haematologic`과 새로운 `R`에서 동일하게 3→4→5를 반복한다. Liver의 score·calibration·실행 폴더를 섞지 않는다.

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
for key in ['model', 'instruction', 'max_tokens', 'temperature', 'workers', 'llm_batch', 'endpoint_sha256', 'llm_server_id']:
    assert len({json.dumps(m[key], sort_keys=True) for m in ms}) == 1, key
assert ms[1]['scores_sha256'] == ms[2]['scores_sha256']
for name, score_time in zip(names, ['test_e5_score_wall_s.txt', 'test_biolord_score_wall_s.txt', 'test_biolord_score_wall_s.txt']):
    summary = read(root / name / 'summary.json')
    rows = read(root / name / 'articles.json')
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

`server-config.txt`와 기존 서버의 설정·운영/캐시 절차를 확인한다. GGUF SHA·바이너리 버전·실제 시작 명령은 확인 가능한 경우 함께 보존하며, 미확인 항목은 동일성을 검증한 것처럼 쓰지 않는다. Live 결과 차이는 4번의 동일 판정을 쓰는 replay와 분리해 LLM 응답 변동도 검토한다.

## 13. 자동 검증과 서버 실험

자동 테스트는 chunk 길이·전체 abstract 보존·E5 prefix·max 집계·무효 점수·threshold 경계·원래 ID·평가 순서·재개·replay·기존 LOTUS 판정 경로를 확인한다. 실제 BioDEX 품질·GPU 시간·모델 효과는 서버에서 측정한다. 이 테스트가 모델 다운로드나 실제 GPU 추론을 대신하지는 않는다.

로컬 경로와 설정의 근거는 [LiteLLM OpenAI 호환 vLLM 경로](https://docs.litellm.ai/docs/providers/vllm), [llama.cpp server 공식 설명](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)이다. 새 설치·다운로드가 필요해도 기존 1·2번의 모델 배포처·revision·파일을 먼저 확인해 같은 가중치를 가져오며 다른 8B 모델로 암묵적으로 바꾸지 않는다.
