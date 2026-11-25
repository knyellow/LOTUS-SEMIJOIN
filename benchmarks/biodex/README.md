# BioDEX 논문 단위 semi-join 실험

BioDEX test 앞 250편에서 간손상 반응 10개 중 하나라도 논문에 나타나는지 기존 `paper` 질문으로 판정합니다. 실험 3·4번은 초록으로 라벨의 평가 순서를 정해 첫 매칭까지 필요한 LLM 호출 수를 비교합니다. 실험 5번은 점수가 낮은 논문을 통째로 생략했을 때 추가로 줄어든 호출 수와 놓친 정답 양성 논문 수를 함께 확인합니다.

| 실험 | 평가 방법 |
|---|---|
| 3 | E5 점수가 높은 라벨부터 평가하고, LLM이 한 쌍을 수락하면 그 논문의 평가를 마칩니다. |
| 4 | 같은 초록 조각을 BioLORD로 채점하고, 3번과 같은 조기 종료 규칙을 사용합니다. |
| 5 | 4번의 점수를 재사용합니다. 모든 라벨 점수가 임계값보다 낮은 논문을 생략하고, 나머지는 4번과 같은 순서로 평가합니다. |

**70B LLM 판정은 기존 GPU 설정을 유지하며, 임베딩은 사전 시험을 통과한 GPU 한 장을 우선 사용합니다.** 사용할 GPU가 없으면 CPU를 선택하고 E5 test·BioLORD test·validation에 같은 장치를 유지합니다. LLM은 세 실험 모두 기존 본문 앞 8,000자를 읽습니다. 새로 실행할 조건은 3번, 4번, 5번의 보수적 정책입니다. 5번의 두 정책은 저장한 4번 판정으로도 비교하며, 1·2번은 과거 기록을 인용합니다.

- [실험 설계와 코드](NEGATIVE_PRUNING_GUIDE.ko.md)에서 초록을 나누는 기준, 임계값을 정하는 방법, 결과 지표를 확인합니다.
- [서버 실행 절차](NEGATIVE_PRUNING_RUNBOOK.ko.md)에 환경 설정부터 3→4→5 실행, 재개, 결과 집계까지의 명령이 있습니다.
- [기존 70B 실험 기록](HISTORICAL_70B_RESULTS.ko.md)에 1·2번 호출 수·F1·경과시간의 출처와 비교 범위를 정리했습니다.

<details>
<summary>기존 BioDEX Join·JoinAndRerank 파이프라인 안내</summary>

아래는 원래 BioDEX 벤치마크의 사용법입니다. RP@5·RP@10은 JoinAndRerank의 순위 지표이며, 위 팀 실험에서는 논문 단위 F1·recall을 사용합니다.

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

</details>
