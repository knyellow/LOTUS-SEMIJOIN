# Abstract 전체가 LLM 입력에 포함되는지 재현하기

검사 대상은 BioDEX test의 모든 4,249편이다. 각 논문의 **abstract 원문 전체가 `fulltext_processed[:8000]`에 연속된 문자열로 포함되는지** 검사한다. 공백·대소문자·구두점 등을 정규화하지 않는다.

```python
bool(abstract.strip()) and abstract in fulltext_processed[:8000]
```

`strip()`은 빈 초록을 거르기 위해서만 사용한다. 실제 포함 검사는 원문 그대로다. Python에서 빈 문자열은 어떤 문자열에도 포함되므로 빈 초록을 성공으로 세지 않는다.

## 재현 명령

저장소 루트에서 Python 3.10~3.12로 실행한다. LOTUS·PyTorch·임베딩 모델·LLM 서버·API 키는 필요하지 않다. 기존 실험 환경을 유지하려면 별도 가상환경에서 아래 두 패키지만 설치해도 된다.

```bash
python -m pip install huggingface-hub==0.36.0 pyarrow==22.0.0
python benchmarks/biodex/audit_abstract_containment.py --output-dir abstract-check-output --check-reference
```

위 한 줄 명령은 Linux와 Windows PowerShell에서 공통으로 사용할 수 있다. 출력은 증거 자료 폴더와 다른 경로를 사용한다.

스크립트는 아래 **고정 revision의 test Parquet 한 파일**을 자동으로 내려받아 SHA-256을 검사한다. 약 107MB이며 이미 캐시에 있으면 재사용한다. `--cache-dir PATH`로 다운로드 캐시 위치를 정할 수 있고, 다운로드 후에는 `--local-files-only`로 네트워크 없이 다시 실행할 수 있다.

| 항목 | 고정값 |
|---|---|
| Dataset | `BioDEX/BioDEX-Reactions` |
| Revision | `01a5dacdabd144a120af04931a11a99febd48432` |
| File | `data/test-00000-of-00001-2449439e9130546c.parquet` |
| File SHA-256 | `7ea93550ba12993ad87d47d0afd665574d42ec14218bca1bfdba13fe3d4ff749` |
| Test 행 수 | 4,249 |

출처는 [고정된 BioDEX 데이터 버전](https://huggingface.co/datasets/BioDEX/BioDEX-Reactions/tree/01a5dacdabd144a120af04931a11a99febd48432)이다. 데이터 원문을 이 저장소에 복제하지 않는다.

## 실행 결과와 확인 방법

| 확인 범위 | 전체 abstract가 앞 8,000자에 포함 | 가장 늦은 abstract 끝 위치 |
|---|---:|---:|
| Test 앞 250편 | 250 / 250 | 3,666 |
| 전체 test | 4,249 / 4,249 | 4,339 |
| 앞 250편 중 liver 정답 양성 | 10 / 10 | 2,693 |
| 앞 250편 중 haematologic 정답 양성 | 32 / 32 | 3,230 |

앞 250편은 데이터셋 순서대로의 첫 250편이다. 무작위 추출이 아니다. 빈 abstract·빈 본문·8,000자를 넘는 abstract·초록이 8,000자에서 잘린 경우는 모두 0편이다. 끝 위치는 Python 문자열의 **0-based exclusive 인덱스**이며, 문자가 4,339개까지인 범위 안에서 초록이 끝난다는 뜻이다. UTF-8 바이트 수나 토큰 수가 아니다.

- [summary.json](summary.json): 데이터 식별자·파일 해시·baseline에서 읽은 길이 제한과 라벨·범위별 집계·CSV 해시.
- [per_article.csv](per_article.csv): **4,249편 전부**의 원래 행 번호·PMID·초록/본문 길이·초록 시작/끝 위치·포함 여부·검사 상태·라벨 패널 양성 여부.

`--check-reference`는 새로 생성한 summary와 CSV 전체를 이 PR의 증거 자료와 대조한다. CSV는 운영체제의 줄바꿈 차이만 허용하며 문자와 행 순서는 그대로 비교한다. 성공하면 `Reference comparison passed`와 종료 코드 0을 반환한다. 파일 해시·행 수·참조 결과가 다르거나 포함되지 않은 논문이 하나라도 있으면 0이 아닌 코드로 종료한다. 실패 시 CSV의 `status`와 `exact_prefix=False` 행으로 해당 PMID를 확인할 수 있다.

검사 범위·라벨은 [기존 baseline](../../semijoin_baseline.py)의 `TRUNCATION_LIMIT`, `LIVER_REACTIONS`, `HAEMATOLOGIC_REACTIONS`를 AST로 읽는다. 따라서 LOTUS를 import하거나 실제 모델을 실행하지 않는다. 길이 제한이나 라벨 설정을 바꾸면 참조 결과와 불일치하므로 결과를 다시 확인해야 한다.

빈 초록·누락 텍스트·정확히 8,000자에서 끝나는 경우와 한 글자 초과한 경우·대소문자/공백 차이는 아래 오프라인 테스트로 확인한다.

```bash
python -m unittest discover -s tests -p test_abstract_containment.py -v
```

## 이 결과로 말할 수 있는 범위

**확인한 revision의 test 전체에서 proxy에 사용하는 abstract 텍스트는 코드가 구성하는 LLM 입력에도 모두 포함된다.** 따라서 proxy에만 추가 abstract 텍스트를 제공해서 얻는 이득이라는 설명은 해당 데이터에서 성립하지 않는다.

이 검사는 코드가 만드는 입력 문자열을 확인한다. 서버가 프롬프트를 추가로 잘라내는지, 모델이 정보를 충분히 활용하는지, 사전학습 지식이 같은지, pruning에 false negative가 없는지는 검증하지 않는다. Train·validation이나 다른 데이터 버전으로도 일반화하지 않는다.
