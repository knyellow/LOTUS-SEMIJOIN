# 서버 AI에게 전달할 실험 실행 요청

PR과 문서를 읽고 사전 점검 → 3번 → 4번 → 5번 → 결과 검증 → 지속 저장소 백업까지 진행해주세요. 실행 중인 작업은 끝까지 추적하고, 실제로 진행을 막는 문제가 있을 때만 보고해주세요.

## 읽을 자료와 기준 코드

- 저장소: https://github.com/knyellow/LOTUS-SEMIJOIN
- PR: https://github.com/knyellow/LOTUS-SEMIJOIN/pull/1 — 접힌 설명까지 읽으세요.
- 브랜치: `feat/biolord-negative-pruning`
- 사용할 커밋: **이 문서를 포함한 PR #1의 커밋**입니다. 준비를 시작할 때 PR의 HEAD와 로컬 HEAD를 대조하고, 전체 SHA를 기록한 뒤 고정하세요.
- 같은 커밋의 [서버 실행 절차](NEGATIVE_PRUNING_RUNBOOK.ko.md), [코드·설계·지표 설명](NEGATIVE_PRUNING_GUIDE.ko.md), [기존 70B 기록](HISTORICAL_70B_RESULTS.ko.md)을 읽으세요.

**이 요청은 점수 공유·단계 시간 합산·4번 전 validation 보정을 요구하던 이전 요청을 대체합니다.** 먼저 `git status --short`로 작업 폴더를 확인하세요. 실험 코드·문서에 미커밋 변경이나 영향을 줄 수 있는 미추적 파일이 있으면 기존 작업을 보존하고, 확인한 PR 커밋의 깨끗한 별도 체크아웃에서 진행하세요. HEAD가 같아도 수정 파일이 남으면 다른 코드가 실행됩니다. Runbook 처음의 fetch/switch/pull은 최초 준비 때만 적용하고, 사용한 문서와 코드가 같은 커밋인지 확인하세요. PR이 갱신돼 읽었던 문서와 HEAD가 달라졌다면 실행 전에 해당 커밋의 문서를 다시 읽으세요. 기준 커밋과 패키지를 기록한 뒤 실험 중 코드를 갱신하거나 벤치마크 의미를 수정하지 마세요.

이미 확인한 PR SHA의 깨끗한 detached worktree를 사용하는 경우에는 Runbook 안내대로 fetch/switch/pull 세 줄을 생략하고, HEAD·미커밋 변경 검사를 마친 뒤 설치부터 진행하세요.

## 고정할 조건

| 항목 | 조건 |
|---|---|
| 과거 1·2번 | 다시 실행하지 않습니다. 보고된 70B 호출 수·F1·시간을 출처와 한계가 있는 별도 역사 표로 인용합니다. |
| LLM | 기존 Llama-3.1-70B-Instruct Q4_K_M GPU 서비스. 실제 모델 ID·가중치·접속 주소·시작 설정을 확인합니다. |
| 판정 | test 앞 250편, liver 라벨 10개와 순서, `PROMPT=paper`, 본문 앞 8,000자, temperature 0, 최대 출력 256토큰, batch·worker 8 |
| Proxy | 전체 abstract, 공통 128토큰 이내 조각, 겹침 없음, 라벨별 최대 cosine similarity. 가중치·정밀도·정규화 유지 |
| 임베딩 장치 | 허가받은 GPU 한 장에서 두 모델 사전 시험 성공 시 GPU, 불가하면 CPU. 3번 전에 한 번 선택해 네 score 계산에서 같은 장치·물리 UUID 유지 |
| 5번 정책 | Validation 양성 100% 보존 conservative 주 조건, 최소 95% retain_95 보조 조건. live는 conservative만 한 번 |

70B GPU 배정·서버·캐시 절차를 임베딩 때문에 바꾸지 마세요. 중지돼 있으면 기존 배포의 원래 기동 절차를 따르며, 다른 사람의 서비스를 중단하거나 8B·다른 양자화로 대체하지 마세요. Test 결과를 보고 threshold 규칙이나 정책을 바꾸지 마세요.

## 1. 사전 준비

Runbook의 서버 준비 절을 순서대로 실행하세요. 기존 LOTUS Python 환경을 활성화하고 `PYTHONOPTIMIZE`를 해제하세요. 같은 사용자·캐시에서 test·validation 데이터, 두 모델 가중치·tokenizer 파일을 미리 준비합니다. 준비된 파일과 기동된 LLM 서비스에서 출발하는 시간을 비교합니다.

새 결과 폴더 `R`을 한 번 만들고 기준 커밋·패키지·데이터 fingerprint·서버 설정·GPU·캐시 절차를 기록하세요. `LLM_MODEL`은 확인한 70B 서비스의 `hosted_vllm/<실제 ID>`로 지정하고 8B 코드 기본값에 의존하지 마세요. 배치 8개와 본문·프롬프트·출력을 수용하는 슬롯당 context를 확인하세요. LOTUS 짧은 요청은 연결과 출력 길이 인자를 포함한 요청의 수락·응답만 확인합니다. 서버가 256토큰 상한을 실제로 적용하는지는 설정·버전·로그로 별도 확인하고, 확인되지 않으면 미확인으로 기록하세요.

과거 기록의 70B 사용 V100 4장과 새 안내의 설치 V100 8장을 구분하세요. 시스템 RAM 48/96GB를 GPU VRAM으로 해석하지 마세요. 가중치·빌드·데이터 revision·캐시의 과거 동일성을 확인할 수 없다면 미확인으로 기록하고, 과거 대비 속도배수를 확정하지 마세요.

지속 저장소 백업 경로·용량과 노드 예약 시간을 미리 확인하세요. `/tmp`나 예약 종료 시 사라지는 디스크만 백업으로 사용하지 마세요. 유지되는 세션/스케줄러에서 진행하고 연결이 끊기면 기존 작업의 생존 여부를 확인해 중복 제출을 피하세요. 인증 값은 기록·백업하지 마세요.

## 2. 임베딩 장치 선택

Runbook의 장치 점검을 그대로 따르세요. Score를 실행하는 호스트의 원래 작업에 배정된 전체 UUID·가시 목록·GPU별 여유·프로세스를 확인합니다. `nvidia-smi`에 보인다는 이유만으로 사용 권한이 있다고 판단하지 마세요.

허가받은 여유 GPU가 있으면 전체 UUID를 후보로 지정하고, 원래 PyTorch 가시 목록에 속하는지 먼저 대조합니다. E5와 BioLORD를 각각 새 프로세스에서 실제 배치로 시험하세요. 두 GPU 시험이 성공하면 GPU를 사용하고, 불가하면 두 CPU 시험이 성공해야 진행합니다. 공통 데이터·모델 파일 오류를 GPU 문제로 숨기지 마세요.

선택 장치·UUID·호스트·원래 가시 목록·이유와 시험 로그를 보존하세요. GPU 마스크는 score 자식 프로세스에만 적용합니다. **E5 test 1회, BioLORD test 2회·validation 1회 모두 같은 장치·UUID**입니다. 본 측정 도중 장치를 바꾸지 말고 GPU 오류가 나면 중단·보존·보고하세요. 새 장치와 새 R로 추가 실측이 필요하면 먼저 팀에 보고합니다.

## 3. 독립된 3→4→5 실측

설정 파일과 최초 장치 선택 기록을 준비한 뒤, Runbook의 `benchmarks/biodex` 작업 위치에서 다음을 차례로 실행하세요. 각 명령은 전용 workflow 폴더에 성공/실패 상태와 전체 외부 시간을 기록합니다. 출력이 있으면 덮어쓰지 않고 중단합니다.

```bash
bash run_semijoin_experiment.sh 3 "$R"
cat "$R/experiment3_workflow/status.json"
bash run_semijoin_experiment.sh 4 "$R"
cat "$R/experiment4_workflow/status.json"
bash run_semijoin_experiment.sh 5 "$R"
cat "$R/experiment5_conservative_workflow/status.json"
```

| 실험 | 한 번의 전체 타이머 안에서 수행 | 고유 test 점수 |
|---|---|---|
| 3 | E5 test score → 정렬·LLM 조기 종료 | `test_e5.json` |
| 4 | BioLORD test score → 정렬·LLM 조기 종료. **Validation·보정 없음** | `test_biolord.json` |
| 5 | BioLORD validation 전체 score → 두 threshold 보정 → **자체 BioLORD test score** → conservative pruning·LLM | `test_biolord_exp5.json` |

100%·95%는 validation 정답 양성 중 LLM 평가 대상으로 남기는 비율입니다. 최종 LLM recall이나 test 보존율의 보장은 아닙니다. Test 정답 양성이 10편인 이번 비교에서는 보존율과 함께 `9/10편`처럼 실제 편수도 보고하세요.

5번은 4번의 점수나 판정을 실행 입력으로 재사용하지 않습니다. 5번의 validation·보정 시간까지 5번 전체 시간에 포함합니다. 규칙은 미리 정한 100%/95%이며 실제 수치는 5번에서 validation만으로 계산합니다. 두 수치가 같아도 정상입니다.

각 실행의 종료 코드·상태·점수 입력 검사·`summary.json`·`articles.json`을 확인한 뒤 다음으로 진행하세요. 새 LLM test 실측은 3회뿐입니다. 전부 pruning되면 5번 progress가 없어도 정상입니다.

## 4. 실측 후 replay와 공통 집계

세 실측을 마친 뒤 Runbook의 replay와 report 명령을 실행합니다. 이 분석은 전체 타이머 밖입니다.

- 두 replay는 **원본 4번의 `test_biolord.json` + 완료 `experiment4/` + 5번 `thresholds.json`**을 사용합니다. 보조 retain_95의 별도 live 실행은 하지 마세요.
- **출력 경로 보호:** `calibrate`·`replay` CLI에는 다른 기존 파일을 출력으로 지정했을 때 덮어쓸 수 있는 알려진 제약이 있습니다. Runbook의 고정 파일명과 `fresh` 검사를 유지하고, 각 명령의 실제 출력 경로가 비어 있는지 먼저 확인하세요. 이번 실행에서 코드를 수정하거나 기존 결과를 덮어쓰지 마세요.
- Report는 3번에 `test_e5.json`, 4번에 `test_biolord.json`, **5번에 `test_biolord_exp5.json`**을 짝지어 줍니다. 과거 원본이 없는 1·2번을 가짜 파일로 만들지 마세요.
- 4·5번 독립 점수 파일은 `embedding_time_s`만 달라도 허용하며 나머지는 모두 같아야 합니다. 실제 점수가 조금이라도 다르면 결과를 보존하고 정렬·gate·원인을 보고하세요. 허용 오차를 임의로 추가하거나 파일을 서로 바꿔 검사를 통과시키지 마세요.

## 5. 지표와 시간 검토

| 보고할 값 | 읽을 위치·주의 |
|---|---|
| 논문 단위 품질·선택 수 | `comparison.json`의 `results[].vs_labels` TP/FP/FN/precision/recall/f1, `results[].n_selected` |
| 논리 LLM 비용 | `lm_calls`. HTTP 재시도·토큰 수와 다릅니다. Replay는 모의 비용이며 추가 실제 호출 0회 |
| 순서 효과 | 첫 gold 순위와 첫 LLM 수락 순위를 구분. 미매칭 K회·생략 0회를 순위로 넣지 않으며 LLM 비교는 공통 ID·편수 표시 |
| Pruning 손실 | Live `summary.json` 또는 replay JSON의 `summary`에서 `n_positive_pruned`·`positive_retention`, 공통 `results[].vs_labels`에서 recall·f1을 읽음. 추가 FN은 양쪽 `articles`의 선택 gold ID 차집합으로 계산. Replay/live를 구분 |
| 주 실행시간 | 각 `실험명_workflow/total_wall_s.txt`. **단계별 시간을 더한 값으로 대체하지 않음** |
| 보조 시간 | 각 score/calibrate/run `*_wall_s.txt`. 내부 `embedding_time_s`·`invocation_time_s`와 측정 범위 구분 |
| Null 요약 | `test_e5_score.log`, `test_biolord_score.log`, `validation_biolord.log`, `test_biolord_exp5_score.log`의 `null_scores=` 개수·영향 논문 수·qids |

Runbook의 시간·순위 확인 Python을 실행하세요. 상태 `complete`, 종료 코드 0, 새 묶음의 모든 단계 완료이며 재개하지 않은 경우만 주 시간을 사용합니다. 실패한 GNU time에도 숫자가 남을 수 있습니다. 일부 단계 재개·LLM만 복구한 시간은 전체 시간으로 쓰지 마세요. `summary.resumed=false`라도 이전 점수를 가져온 새 LLM 실행은 전체 실측이 아닙니다.

주 시간에는 각 방법의 score·LLM 처리와 입출력·프로세스·묶음 내부 검증, **5번만 필요한 validation·보정**이 포함됩니다. 최초 다운로드·70B 기동·GPU/CPU 사전 시험·연결 점검·실측 후 replay/report는 제외합니다. 측정 중 추가 다운로드가 발생했다면 표시하세요.

보고서는 3 live + 2 replay가 정상이고 `missing_experiments=[1,2]`, `vs_join_full=null`이 정상입니다. 과거 1·2의 P/R은 보고된 값이 없어 별도로 추정·인용하지 않습니다. 전체 선택 ID가 없어 선택 일치도·동일 ID witness는 복원할 수 없다고 적으세요. 과거 외부 시간과 현재 범위·서버·장비·캐시 차이가 불명확하면 과거는 참고값, 동일 현재 환경의 3~5를 주 시간 비교로 사용합니다. Validation 양성 100% 보존을 test 무누락 보장으로 설명하지 마세요.

## 6. 실패·재개와 백업

실패하면 다음 단계로 넘어가지 말고 최초 로그·workflow status·타이머·점수·progress를 보존하세요. 같은 명령 재호출로 파일을 덮지 마세요. LLM만 복구할 때는 Runbook 재개 절과 원래 점수·threshold를 사용하고 전체 시간은 미측정으로 남깁니다. Workflow 상태를 수동으로 complete로 바꾸지 마세요. 추가 live나 조건 변경이 필요하면 먼저 상황을 보고하세요.

`R` 전체를 지속 저장소로 복사하고 파일 목록·크기·SHA-256을 대조하세요. 네 score JSON, threshold, live 세 폴더, 두 replay, comparison, 모든 단계와 workflow의 시간·로그·상태, 장치 선택·시험·환경·서버 기록이 들어가야 합니다. 인증 값과 모델 가중치 자체는 결과 백업에서 제외합니다. 원본은 백업 검증 전에 삭제하지 마세요.

## 최종 보고

실제 코드 커밋, 서버·장치와 과거 미확인 조건, 단계별 성공/실패와 오류, 3·4·5 품질·호출 수·전체 시간·5번 준비 비용, replay 두 조건의 추가 FN·절감, 네 null 요약을 표로 정리해주세요. 결과·로그 원본 경로와 검증된 지속 백업 경로·SHA 대조 결과, 재개 여부·시간 미측정 항목·남은 제한도 적어주세요. 실제로 실행하지 못한 작업은 완료로 표시하지 마세요.
