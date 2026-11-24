> 팀 semi-join 실험 3·4·5는 아래 upstream JoinAndRerank 실행과 다릅니다. 로컬 경량 LLM을 쓰는 [한국어 가이드 11절](NEGATIVE_PRUNING_GUIDE.ko.md#11-서버-명령-3--4--5-순서대로)을 따라 진행하세요.

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

전체 사전 설정·로컬 서버 명령·복사 가능한 실험별 명령은 [한국어 코드 설명과 실행 가이드](NEGATIVE_PRUNING_GUIDE.ko.md)에 정리했습니다. **11.0 공통 준비 → 11.1 실험 3 → 11.2 실험 4 → 11.3 실험 5** 순서입니다.

| 순서 | 실행 전 준비 | 실행 | 확인할 결과 |
|---|---|---|---|
| 공통 | 기존 1·2번 Python 환경·같은 GGUF·llama-server·실행 설정 확인 | 서버는 각 live run 직전에 동일 명령으로 시작 | 가중치 SHA·서버 버전·명령·설정 기록 |
| 3 | 같은 test 논문의 E5 점수, 서버 연결 점검 | `run ... --experiment 3` | 3번 summary·articles·progress |
| 4 | BioLORD test·validation 점수, threshold 고정, 두 모델의 입력·chunk 일치 검사 | `run ... --experiment 4` | 4번 완료 결과와 보존된 BioLORD 점수 |
| 5 | 같은 BioLORD 점수·validation threshold·4번 완료 결과 | 두 정책 replay → `run ... --experiment 5 --calibration ... --policy conservative` | 두 정책 호출 수·품질, 보수적 조건 실제 시간 |

코드 기본값은 `hosted_vllm/llama3.1-8b`와 로컬 `http://127.0.0.1:8080/v1`입니다. 이는 LiteLLM을 통한 OpenAI 호환 경로이며 실제 서버는 llama.cpp `llama-server`를 사용할 수 있습니다. 코드가 LLM 가중치를 다운로드하거나 서버를 시작하는 것은 아닙니다. 과거 N=250의 정확한 가중치·양자화·서버 설정은 저장소만으로 확정하지 못하므로 **기존 1·2번의 파일과 설정을 먼저 확인하고 재사용**합니다. 외부 OpenAI 유료 모델로 자동 전환하지 않습니다.

Proxy는 abstract 전체를 두 tokenizer에서 모두 128토큰 이내인 동일한 조각으로 나누며, 논문–라벨 점수는 조각별 cosine similarity의 최대입니다. E5는 문서 `passage: `·라벨 `query: ` 접두어를 쓰고 BioLORD는 접두어 없이 씁니다. **LLM 입력은 기존 `fulltext_processed[:8000]`**입니다. 빈 abstract·무효 embedding은 원래 순서로 평가하고 pruning하지 않습니다.

3·4번에는 threshold를 넣지 않습니다. 5번의 threshold는 validation의 정답과 점수만으로 정해 추가 LLM 호출이 없으며 4번 test 결과 전에 고정합니다. `conservative`를 주 조건, `retain_95`를 보조 조건으로 보고합니다. Validation 보존율이 test의 무누락을 보장하지는 않습니다.

같은 비교에서는 `input_sha256`, `chunks_sha256`, `scorer.chunking`, `reactions`가 일치해야 합니다. 4·5번에는 같은 BioLORD 점수 파일을 씁니다. **Score는 덮어쓰지 않고 재개·replay에 그대로 재사용**하며 새 측정에는 새 출력·timer를 사용합니다. 가이드의 `fresh` 검사가 기존 파일을 보호합니다.

응답 cache는 기본적으로 꺼져 있지만 서버 KV cache는 별개이므로 live run마다 같은 서버 초기화·짧은 사전 점검을 적용합니다. `EARLY_WORKERS`는 논문 병렬도, `LLM_BATCH`는 LM 호출 내부 상한, 서버 `--parallel`은 슬롯 수입니다. 본문 8,000자와 프롬프트·출력을 수용할 실제 슬롯당 context를 시작 로그에서 확인합니다.

결과는 `manifest.json`, `progress.jsonl`, `articles.json`, `summary.json`에 남깁니다. `LLM_SERVER_ID`와 manifest 해시는 실행 설정·결과 혼용을 검출하지만 실제 서버 파일을 자동 확인하지는 않습니다. 생략 논문도 F1의 전체 분모에 남기며, 논리적 pair 호출·F1/recall·gold/LLM witness 순위·양성 생략을 기록합니다. `n_positive_pruned`와 4번 대비 추가 FN은 다릅니다.

Replay는 호출 수·품질을 재계산하며 실제 실행시간을 재현하지 않습니다. 시간 비교는 test 점수 생성과 무중단 live run의 외부 wall time을 사용하고 validation 준비 비용을 별도로 표시합니다. 재개·완료 재호출의 짧은 시간으로 원래 timer를 덮어쓰거나 전체 시간이라고 보고하지 않습니다. 코드 호출 경로와 저장 지표의 세부 의미는 가이드 2~10절, 결과 확인 코드는 12절을 참고합니다.
