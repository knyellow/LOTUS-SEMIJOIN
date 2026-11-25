> 팀 semi-join 실험 3·4·5는 아래 upstream JoinAndRerank 실행과 다릅니다. 70B GPU를 쓰는 [한국어 가이드 11절](NEGATIVE_PRUNING_GUIDE.ko.md#11-서버-명령-3--4--5-순서대로)을 따라 진행하세요.

Instructions for running biodex pipeline:

Setup:
```
conda create lotus-bench python=3.10 -y
conda activate lotus-bench
pip install -r requirements.txt
```

Run the biodex tester file which includes the following two programs:
- Join only
- JoinAndRerank (which uses follows the join with listwise re-ranking, resembling the orignal logic in https://arxiv.org/pdf/2401.12178, and comparble to the docetl baseline)
Both are configured with gpt-4o-mini as the LLM and text-embeddings-3-small as the retreiver model

```
python biodex_tester.py
```

This will run both the Join and JoinAndRerank programs and output summary metrics for each, including RP@K and total_latency, in the directory biodex_results.
Note that the latency for the JoinAndRerank pipeline measures the ranking step only, reusing the Join outputs as intermediate results, so latency numbers from the Join should be added to get the latency for the full JoinAndRerank pipeline.


You can find example results I ran earlier in biodex_results/Mar-17-25, which gave the following for the full JoinAndRerank 

```
+-------------------------+--------+---------+-------------+
|                         |   RP@5 |   RP@10 | Latency (s) |
+=========================+========+=========+=============+
| Lotus JoinAndRerank     | .2669  |  .2722  |  566.55     |
+-------------------------+--------+---------+-------------+
```

## Abstract proxy 기반 semi-join: 실험 3 → 4 → 5

**70B GPU에서 3·4·5만 새로 실행합니다.** 1·2번은 과거 70B·paper·간 패널·test 앞 250편 실행의 호출 수, 라벨 F1, 외부 경과시간을 인용합니다. 과거 전체 조인은 2,500회·F1 0.172·7,282초, 고정 순서 조기 종료는 1,855회·F1 0.172·1,896초입니다. [과거 수치의 출처와 측정 범위](HISTORICAL_70B_RESULTS.ko.md)

| 순서 | 준비와 실행 | 확인할 결과 |
|---|---|---|
| 공통 | 현재 사용 가능한 70B GPU 서비스·모델 ID 확인, 코드·장비·캐시 정책 기록, 데이터·proxy 가중치·tokenizer 사전 준비 | 과거 기준은 Llama-3.1-70B-Instruct Q4_K_M, V100 16GB 4장. 현 배포와의 차이 기록 |
| 3 | E5 test 점수 생성 → 정답 양성 인덱스 10개·라벨 순서 대조 → 서버 사전 점검 → 3번 실행 | E5 점수, 완료 결과, 외부 시간 |
| 4 | BioLORD test·validation 점수 → threshold 고정 → 입력·조각 일치 확인 → 서버 사전 점검 → 4번 실행 | 4번 완료 결과와 보존된 BioLORD 점수 |
| 5 | 같은 BioLORD 점수와 4번 완료 결과로 두 정책 replay → 서버 사전 점검 → 보수적 조건 실측 | 두 정책의 호출 수·품질과 보수적 조건의 실측 시간 |

서버의 실제 70B 모델 ID로 `LLM_MODEL=hosted_vllm/<실제 ID>`를 지정하고 `LLM_BATCH=8`, `EARLY_WORKERS=8`을 사용합니다. 일반 코드 기본값은 8B입니다. 가이드의 명령은 기존 OpenAI 호환 서비스에 연결하며 모델 서버를 기동하지 않습니다. [서버 준비와 실험별 전체 명령](NEGATIVE_PRUNING_GUIDE.ko.md#11-서버-명령-3--4--5-순서대로)

Proxy는 abstract 전체를 두 tokenizer에서 모두 128토큰 이내인 동일한 조각으로 나누며, 논문–라벨 점수는 조각별 cosine similarity의 최대입니다. E5는 문서 `passage: `·라벨 `query: ` 접두어를 쓰고 BioLORD는 접두어 없이 씁니다. **LLM 입력은 기존 `fulltext_processed[:8000]`**입니다. 빈 abstract·무효 embedding은 원래 순서로 평가하고 pruning하지 않습니다.

3·4번에는 threshold를 넣지 않습니다. 5번의 threshold는 validation의 정답과 점수만으로 정해 추가 LLM 호출이 없으며 4번 test 결과 전에 고정합니다. `conservative`를 주 조건, `retain_95`를 보조 조건으로 보고합니다. Validation 보존율이 test의 무누락을 보장하지는 않습니다.

같은 비교에서는 `input_sha256`, `chunks_sha256`, `scorer.chunking`, `reactions`가 일치해야 합니다. 4·5번에는 같은 BioLORD 점수 파일을 씁니다. **Score는 덮어쓰지 않고 재개·replay에 그대로 재사용**하며 새 측정에는 새 출력·timer를 사용합니다. 가이드의 `fresh` 검사가 기존 파일을 보호합니다.

응답 cache는 기본적으로 꺼져 있지만 서버 KV cache는 별개이므로 live run마다 같은 기존 운영/캐시 절차와 사전 점검을 적용합니다. 초기 KV 상태를 통제하지 못하면 그 사실을 기록하고 cold-start 속도 비교로 해석하지 않습니다. `EARLY_WORKERS`는 논문 병렬도, `LLM_BATCH`는 LM 호출 내부 상한, 서버 `--parallel`은 슬롯 수입니다. 본문 8,000자와 프롬프트·출력을 수용할 실제 슬롯당 context를 시작 로그에서 확인합니다.

결과는 `manifest.json`, `progress.jsonl`, `articles.json`, `summary.json`에 남깁니다. `LLM_SERVER_ID`와 manifest 해시는 실행 설정·결과 혼용을 검출하지만 실제 서버 파일을 자동 확인하지는 않습니다. 생략 논문도 F1의 전체 분모에 남기며, 논리적 pair 호출·F1/recall·gold/LLM witness 순위·양성 생략을 기록합니다. `n_positive_pruned`와 4번 대비 추가 FN은 다릅니다.

모든 실행이 끝나면 3~5번 완료 결과와 두 replay의 품질을 `comparison.json`으로 집계합니다. 과거 요약·선택 CSV가 없어 `missing_experiments`는 `[1, 2]`, `vs_join_full`은 `null`이 정상입니다. 3~5번의 정답 대비 P/R/F1은 같은 계산식·네 자리 반올림을 사용합니다. 과거 시간과 속도를 비교할 때에는 GPU·서버·캐시 조건의 일치 여부를 표시합니다. [집계 명령과 결과 해석](NEGATIVE_PRUNING_GUIDE.ko.md#121-실험-35의-품질-집계와-과거-12번-인용)

Replay는 호출 수·품질을 재계산하며 실제 실행시간을 재현하지 않습니다. 시간 비교는 test 점수 생성과 무중단 live run의 외부 wall time을 사용하고 validation 준비 비용을 별도로 표시합니다. 재개·완료 재호출의 짧은 시간으로 원래 timer를 덮어쓰거나 전체 시간이라고 보고하지 않습니다. 코드 호출 경로와 저장 지표의 세부 의미는 가이드 2~10절, 결과 확인 코드는 12절을 참고합니다.
