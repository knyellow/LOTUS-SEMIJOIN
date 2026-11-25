# 기존 70B·GPU 실험 1·2번: 재사용할 수치와 한계

이번 계획은 **기존 1·2번의 수치를 출처와 함께 활용하고, 3·4·5번만 새로 실행**하는 것이다. 이 문서는 사용자가 제공한 서버 AI 조사 보고서(2026-11-25)를 요약한다. 이 PR에서 원격 서버의 원본을 직접 확인하거나 실험을 다시 실행한 결과는 아니다.

## 기존 결과 표

| 실험 | 방법 | 논문 수 | 정답 양성 수 | 논리적 LLM 호출 수 | 보고된 article-level F1 | 외부 경과시간 |
|---|---|---|---|---|---|---|
| **1** | Full Join + Projection (`join_full`) | 250 | 10 | **2,500** | **0.172** | **7,282초** |
| **2** | 고정 순서 Early Stopping (`early_stop`) | 250 | 10 | **1,855** | **0.172** | **1,896초** |

F1은 조사 보고서의 표시 정밀도 그대로다. Precision·recall·정수 TP/FP/FN·방법별 전체 선택 ID는 이 발췌만으로 채우지 않는다. F1이 같아도 두 방법의 선택 논문 집합이 같다는 뜻은 아니다. 호출 수는 메시지 단위 논리적 pair 판정이며 HTTP 재시도나 토큰 수가 아니다.

## 보고된 실행 조건

- 데이터: `BioDEX/BioDEX-Reactions`, test 앞 250행, 본문 앞 8,000자, `liver` 10개 라벨, `paper` 질문.
- 정답 양성 행 번호: `[29, 119, 134, 143, 148, 153, 158, 167, 198, 223]`. 새 test score에서 대조하되, 같은 행 번호와 정답 수만으로 과거 데이터 revision까지 확인되는 것은 아니다.
- 모델: `bartowski/Meta-Llama-3.1-70B-Instruct-GGUF`, **Q4_K_M**.
- 장비: **V100 16GB 4장**, `c4130-110533.wisc`.
- 서버 옵션: `-ngl 99 -c 65536 --parallel 8 -fa on -ctk q8_0 -ctv q8_0 --no-jinja --chat-template llama3`.
- 클라이언트: `LLM_BATCH=8`, `EARLY_WORKERS=8`, temperature 0, 최대 출력 256토큰.
- 같은 서버에서 early → full → cascade 순서. 1·2번은 재개 없음, full은 단일 shard, 종료 코드 0으로 보고됨. 초기 서버 KV cache의 동일성은 입증되지 않았다.
- 과거 데이터 revision, 가중치 SHA, llama.cpp 빌드 커밋은 미확인이다. CloudLab 예약이 종료된 것으로 추정되지만 조사 보고서는 노드 접속을 시도하지 않았다. 과거 서비스가 현재도 실행 중이라고 가정하지 않는다.

위 옵션은 **과거 조건의 기록**이다. 현재 서비스의 가중치·장비·설정을 확인하는 기준으로 사용하며, 이 문서의 추정 경로로 서버를 새로 띄우는 명령이 아니다. 실제 서비스 모델 alias와 연결 주소는 현재 서버에서 확인한다.

## 시간의 근거와 비교 범위

조사 보고서는 `run_70b.sh`가 Python 명령 앞뒤에 기록한 아래 발췌를 제시했다. LLM 서버는 먼저 실행돼 있었고, Python 시작부터 데이터 읽기·모델 클라이언트 구성·LLM 판정·결과 저장·종료까지의 외부 시간이다.

```text
=== early start Fri Nov 13 10:58:29 PM CST 2026
=== early exit 0 Fri Nov 13 11:30:05 PM CST 2026
=== full start Fri Nov 13 11:30:05 PM CST 2026
=== full exit 0 Sat Nov 14 01:31:27 AM CST 2026
```

두 차이는 각각 **1,896초**, **7,282초**다. `/usr/bin/time`으로 잰 값이나 기존 summary의 내부 `time_s`가 아니다. 새 실행은 `/usr/bin/time`으로 같은 명령 구간을 측정한다. 도구 이름보다 시작·종료 범위와 실행 조건을 맞추는 것이 중요하다.

3번 시간은 E5 test score 명령 wall time + 3번 run wall time, 4·5번 시간은 같은 BioLORD test score 명령 wall time + 각 run wall time이다. 공유 score를 실제 한 번 생성하더라도 독립 방법 비용에는 각각 1회분을 귀속한다. 최초 파일 다운로드·서버 기동·validation 보정·사전 점검은 별도로 준비·기록한다. 프로그램에서 모델을 메모리에 다시 올리는 시간과 결과 저장은 포함한다. 사전 cache 준비가 온라인 메타데이터 조회까지 없애는 것은 아니며, 본 측정에서 파일 다운로드가 실제 발생하면 네트워크/다운로드 포함 측정으로 표시한다.

**과거와 새 실행의 GPU 종류·장수, 가중치·서버 설정, 동시성, cache/워밍업 조건이 확인되지 않으면 과거 대비 속도배율을 동일 조건의 알고리즘 효과로 해석하지 않는다.** 이때 과거 1·2번 시간은 참고값으로 표시하고, 같은 현재 환경에서 수행한 3·4·5번을 주 시간 비교로 사용한다.

## 원본 부재와 집계 정책

보고서는 과거 `summary_full.json`, `summary_early.json`, 선택 CSV와 전체 진행 기록이 삭제됐다고 설명한다. 남은 것은 세션 기록·작업 출력 발췌와 early 일부 42/250행이다. 원래 결과 경로는 `/root/lotus/benchmarks/biodex/semijoin_results/llama3.1-70b/n=250/`였다. 문서에서 이 경로의 현재 존재를 보장하지 않는다.

- 이번 실행은 과거 파일을 필수 입력으로 요구하지 않는다. `run`·`report`에 `--baseline-summary`를 전달하지 않는다.
- 새 `comparison.json`에는 3·4·5번 live 및 5번 replay를 집계한다. **`missing_experiments=[1,2]`와 `vs_join_full=null`이 정상**이다.
- 위 기존 수치는 별도 역사 결과 표로 사용한다. scalar F1·호출 수로 가짜 원본 summary나 선택 CSV를 만들거나 `report` 검증을 우회하지 않는다.
- 과거 1번 대비 선택 일치도 `vs_join_full`, 과거 1·2번과 같은 ID에서의 witness 순위 비교는 현재 자료만으로 계산하지 않는다. 42행을 전체 250편의 witness 평균으로 사용하지 않는다.
- 새 3·4번의 정렬은 같은 gold 양성 ID의 `first_gold_witness_rank`와 전체 호출 수로 비교한다. 새 4→5의 추가 FN은 새 결과의 선택 gold ID 차이로 계산하며, replay와 별도 live를 구분한다.

과거 원본을 나중에 복구하면 기존 `check-baselines`와 `report --baseline-summary` 기능을 사용할 수 있다. 현재 합의한 기본 경로는 재실행이나 복구를 선행 조건으로 삼지 않는다.

## 출처와 제외한 실행

보고서가 지목한 근거는 `/root/.claude/projects/-root/15b74d0b-1e24-4b8b-b074-77da50243e37.jsonl`, `/tmp/claude-0/-root/15b74d0b-1e24-4b8b-b074-77da50243e37/tasks/{bopev2c51,bicijhru2,bu4qd49hz}.output`, 같은 세션 scratchpad의 `run_70b.sh`다. 시간 발췌의 원래 노드 파일은 `/mydata/progress.txt`다. `/tmp` 자료는 휘발성이 있으므로 서버에서 접근 가능하면 관련 실험 근거만 별도로 보존한다. 개인 대화 전체나 인증 정보는 공개하지 않는다.

**이번 기준에서 제외:** 8B·CPU full, 8B의 예상 early 호출 1,285회, cascade 2,115회, 70B·ADR·1,000편 결과, n=25/n=35 점검 결과. 회의록의 `cloudlab_node/semijoin_250.log`는 context 초과로 중단된 별도 실행으로 보고돼 최종 70B 결과의 근거로 사용하지 않는다.
