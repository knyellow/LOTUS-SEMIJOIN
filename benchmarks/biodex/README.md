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

## Abstract proxy 기반 semi-join: 실험 3·4·5

[한국어 코드 설명·지표·전체 서버 명령](NEGATIVE_PRUNING_GUIDE.ko.md)에서 작은 코드 단위와 호출 경로를 확인할 수 있습니다.

| 실험 | Proxy 모델 | Pruning |
|---|---|---|
| 3 | E5-base-v2 | 없음 |
| 4 | BioLORD-2023 | 없음 |
| 5 | BioLORD-2023 | 최대 라벨 점수가 threshold 미만이면 논문 전체 생략 |

Proxy는 **abstract 전체를 같은 조각으로 나누어** 임베딩합니다. 각 조각은 E5·BioLORD 양쪽 tokenizer에서 special token과 E5 `passage: ` 접두어까지 포함해 128토큰 이내입니다. 조각은 겹치지 않고 원문 전체를 보존합니다. 논문–라벨 점수는 각 조각의 정규화 cosine similarity 중 최댓값입니다. E5 라벨에는 `query: ` 접두어를 붙이고 BioLORD에는 접두어를 사용하지 않습니다. LLM 입력은 모든 조건에서 기존 **`fulltext_processed[:8000]`**입니다.

기존 LOTUS `SentenceTransformersRM`과 팀의 pairwise 조기 종료 실행기를 재사용합니다. 아래는 `benchmarks/biodex`에서 실행하는 최소 예시입니다. 먼저 `python -m pip install -e ../.. datasets`로 현재 checkout을 설치하고 LLM 서버를 실행합니다. `REACTION_SET`, `PROMPT`, `LLM_MODEL`, `LLM_API_BASE`, `LLM_BATCH`, `EARLY_WORKERS`는 비교군과 동일하게 맞춥니다. 혈액 패널은 `REACTION_SET=haematologic`과 별도 출력 경로를 사용합니다.

```bash
export REACTION_SET=liver
R=pruning_results/liver_paper
mkdir -p "$R"

# 준비: validation BioLORD + threshold, test E5/BioLORD. LLM 호출 0회.
python semijoin_pruning.py score --model biolord --split validation --device cuda \
  --output "$R/validation_biolord.json"
python semijoin_pruning.py calibrate "$R/validation_biolord.json" --output "$R/thresholds.json"
python semijoin_pruning.py score --model e5 --split test --n-articles 250 --device cuda \
  --output "$R/test_e5.json"
python semijoin_pruning.py score --model biolord --split test --n-articles 250 --device cuda \
  --output "$R/test_biolord.json"

# 3·4번: threshold 없이 모든 논문을 정렬된 순서로 평가.
python semijoin_pruning.py run "$R/test_e5.json" --experiment 3 --output-dir "$R/experiment3"
python semijoin_pruning.py run "$R/test_biolord.json" --experiment 4 --output-dir "$R/experiment4"

# 5번의 두 threshold: 완료된 4번 결과로 재생. 추가 LLM 호출 0회.
python semijoin_pruning.py replay "$R/test_biolord.json" --calibration "$R/thresholds.json" \
  --policy conservative --source-dir "$R/experiment4" --output "$R/experiment5_conservative_replay.json"
python semijoin_pruning.py replay "$R/test_biolord.json" --calibration "$R/thresholds.json" \
  --policy retain_95 --source-dir "$R/experiment4" --output "$R/experiment5_retain95_replay.json"

# 5번 실제 실행: 선택한 정책의 wall time은 이 명령을 별도로 실측.
python semijoin_pruning.py run "$R/test_biolord.json" --experiment 5 \
  --calibration "$R/thresholds.json" --policy conservative --output-dir "$R/experiment5_conservative"
```

CPU 임베딩은 `--device cpu`로 지정합니다. 위 test 앞 250편을 바꾸면 E5·BioLORD 양쪽에 같은 개수를 사용합니다. 모델별 test 점수의 `input_sha256`, `chunks_sha256`, `scorer.chunking`, `reactions`가 일치해야 합니다. 전체 `scorer`의 모델·접두어는 두 모델에서 다른 것이 정상입니다. 4·5번에는 같은 BioLORD 점수 파일을 사용합니다. 기존 본문 prefix 점수와 calibration은 새 조각 표현으로 다시 생성해야 합니다.

Calibration은 validation의 정답 라벨과 proxy 점수만 사용하므로 4번 LLM 실행을 먼저 할 필요가 없습니다. `conservative`는 validation 정답 양성을 모두, `retain_95`는 최소 95%를 유지하는 경계를 선택합니다. 적은 양성 수나 동점 때문에 두 값이 같아질 수 있으며 test 보존율을 보장하지는 않습니다. `--threshold NUMBER`로 미리 정한 값을 전달할 수도 있습니다. Threshold와 같은 점수·무효 점수·빈 abstract는 생략하지 않으며 proxy가 무효이면 원래 라벨 순서로 LLM 평가합니다.

출력은 `manifest.json`, `progress.jsonl`, `articles.json`, `summary.json`입니다. 생략된 논문도 전체 결과에 남기며 정답 양성이면 F1의 FN에 포함합니다. Article-level precision·recall·F1, gate의 양성 보존율, 생략한 양성 수, 논리적 pair 호출 수, 첫 LLM 수락 순위를 저장합니다. 첫 정답 라벨 순위는 논문별로 별도 기록합니다. `n_positive_pruned`는 4번 대비 추가 누락과 다릅니다. 실제 HTTP 재시도 수나 미완료 논문의 호출은 완료 기록의 합계에 포함되지 않습니다.

Replay는 같은 점수 파일을 사용한 완료 실행의 진행 기록·결과·요약을 검증합니다. 이전에 생략한 논문을 살리려는데 원본 판정이 없으면 중단합니다. Replay의 호출 수·F1은 정책 비교에 쓰되 실제 실행시간으로 해석하지 않습니다. `estimated_embedding_plus_run_time_s`는 일부 임베딩 준비·점수 구간과 run 구간을 더한 추정치이며 재개하면 전체 추정치를 제공하지 않습니다. 전체 명령 wall time은 가이드의 `/usr/bin/time` 예시로 측정합니다.

같은 출력 폴더는 설정이 완전히 같은 재개에만 사용하며 한 프로세스만 기록합니다. 로그·타이머 파일은 출력 폴더 바깥에 저장합니다. 공통 LOTUS pair 판정과 parser는 기존 baseline 동작을 유지합니다.
