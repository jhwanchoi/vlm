# stress — 서빙 부하 테스트

설계 근거: [docs/superpowers/specs/2026-07-30-vlm-stress-test-design.md](../docs/superpowers/specs/2026-07-30-vlm-stress-test-design.md)

측정 목표는 네 가지다. 포화점(knee), 동시성별 지연 곡선, 파괴점, DP=2 확장 효율.

## 최우선 제약: 타 팀 무침범

성능 수치보다 우선한다. 직접 경로는 플래그로 막고 간접 경로는 감시로 막는다.

| 자원 | 조치 |
|---|---|
| CPU | `--cpuset-cpus=12-15,44-47` (dst+dmt 배정 `0-15,32-47` 의 부분집합) |
| NUMA | `--cpuset-mems=0` |
| RAM | 컨테이너당 `8g`. 기존 서빙 130g 과 합쳐 몫 225g 이내 |
| GPU | `--gpus` 미지정. 부하기는 GPU 를 인식하지 못한다 |
| 포트 | 8089-8093 (8xxx 규칙) |
| 대상 | `:8000`, `:8001`, `:8002` 만. `locustfile.py` 가 코드로 검사한다 |

`run.sh preflight` 가 위 항목을 전부 확인하고, 하나라도 어긋나면 기동하지 않는다.

`monitor.sh` 는 부하기와 **분리된 프로세스**로 돌며 다음 조건에서 부하를 자동 중단한다.
부하기가 죽어도 감시와 중단 권한이 남아야 하기 때문이다.

| 트리거 | 임계 |
|---|---|
| 타 팀 GPU(4-7) 온도 | 83도 초과 |
| 타 팀 GPU(4-7) SM 클럭 | 기준선 대비 15% 하락 30초 지속 |
| NUMA node0 여유 | 8 GiB 미만 |
| load average | 96 초과 |
| LB 5xx 비율 | 20% 초과 30초 지속 (dst 영향 감지) |

## 준비

이미지 생성은 Pillow 가 필요해서 로컬에서 한다 (서버에 pip 이 없다).

```bash
python3 stress/make_assets.py          # assets/img_1mp.jpg, img_native.jpg
rsync -a stress/ dmt@<서버>:/home/dmt/local-vlm/stress/
```

서버에서 부하기 이미지 준비:

```bash
docker pull locustio/locust:2.32.4
```

## 실행

```bash
cd /home/dmt/local-vlm/stress

./run.sh preflight run1 8089 http://host.docker.internal:8000   # 검사만
./run.sh run       run1 8089 http://host.docker.internal:8000   # 주 측정 (15분)

# DP=2 확장 효율 (레플리카 1개 직결, 32까지)
STAGES="1,2,4,8,12,16,24,32" ./run.sh run run2 8090 http://host.docker.internal:8001

# 해상도별 비용 (시나리오 단독)
STAGES="8,16" SCENARIOS=S1 ./run.sh run run3a 8091 http://host.docker.internal:8000
STAGES="8,16" SCENARIOS=S2 ./run.sh run run3b 8092 http://host.docker.internal:8000
STAGES="8,16" SCENARIOS=S3 ./run.sh run run3c 8093 http://host.docker.internal:8000
```

화면 확인은 로컬에서 터널을 열고 브라우저로 본다.

```bash
ssh -L 8089:localhost:8089 -L 8090:localhost:8090 dmt@<서버>
# http://localhost:8089
```

컨테이너는 테스트가 끝나도 내리지 않는다. Web UI 에서 결과를 계속 볼 수 있고,
수동으로 재실행해도 `monitor.sh` 가 대기 상태로 남아 가드가 걸린다.

## Locust UI 데이터 내보내기

UI 의 통계는 컨테이너 메모리에만 있다. 컨테이너를 내리면 사라지므로 내보내 둔다.
테스트 종료 후 부하를 다시 걸지 않았다면 종료 시점 스냅샷이다.

```bash
./export.sh          # 떠 있는 모든 vlm-stress-* 컨테이너
./export.sh run1     # 특정 런만
```

각 런의 결과 디렉토리 아래 `export/` 에 저장된다. UI 탭 구성과 같다.

| 파일 | UI 대응 |
|---|---|
| `statistics.csv` | Statistics (시나리오별 요청 수, 지연 분위, 평균 콘텐츠 크기 = 토큰 수) |
| `charts_report.html` | Charts (RPS, 응답시간, 사용자 수 시계열 포함) |
| `failures.csv` | Failures |
| `exceptions.csv` | Exceptions |
| `ratio.json` | Current ratio (시나리오 가중치) |
| `locust_logs.json` | Logs |
| `stats_snapshot.json` | 종료 시점 상태 전체 스냅샷 |
| `container.log` | 컨테이너 표준 출력 (warmup 결과, shape 전환 이력) |

## 결과

실행 중에는 서버 `/data01/dmt/stress-results/<타임스탬프>-<이름>/` 에 쌓이고,
측정이 끝나면 [results/](results/) 로 레포에 담는다. Locust 컨테이너를 내리면
`export/` 내용은 서버에서 사라지므로(UI 통계가 메모리에만 있다) 레포가 보관 위치다.

2026-07-30 실측 원자료는 [results/](results/) 에 있고 해석은 [docs/09](../docs/09-stress-test-results.md) 다.

| 파일 | 내용 |
|---|---|
| `summary.md` | 동시성별 표, 판정(knee/파괴점/실질 상한/권고 동시성), 무침범 증빙 |
| `report.html` | Locust HTML 리포트 |
| `locust_stats*.csv` | 클라이언트 측 원자료 |
| `engine.csv` | 레플리카별 엔진 지표 1초 샘플 |
| `gpu.csv`, `host.csv` | GPU 8장, load average, NUMA 여유 1초 샘플 |
| `guard.log` | 가드 판정 이력 |
| `meta.json` | 실행 구성, 시작 시점 컨테이너와 GPU 스냅샷 |

집계는 단독 재실행할 수 있다.

```bash
python3 analyze.py /data01/dmt/stress-results/<디렉토리> > /tmp/summary.md
```

## 구성 요소

| 파일 | 책임 |
|---|---|
| `locustfile.py` | 요청 1건의 전송과 계측 (SSE 첫 청크로 TTFT). 부하 형태는 모른다 |
| `shapes.py` | 시간에 따른 동시성만. 요청 내용은 모른다 |
| `monitor.sh` | 관측과 자동 중단만. 부하기 내부는 모른다 (REST API 로만 통신) |
| `run.sh` | 오케스트레이션, preflight, watchdog, 결과 수집 |
| `analyze.py` | CSV 입력, `summary.md` 출력. 표준 라이브러리만 사용 |
| `make_assets.py` | 합성 이미지 생성 (로컬 실행, Pillow 필요) |
| `export.sh` | Locust UI 데이터 내보내기 (컨테이너를 내리기 전에 실행) |

## 워크로드

| 시나리오 | 비중 | 입력 | 이미지 토큰 |
|---|---|---|---|
| S1 | 2 | 텍스트만 | 0 |
| S2 | 5 | 1024x1024 | 약 1.0K |
| S3 | 3 | 4464x2160 | 약 9.4K |

- 출력은 `max_tokens=256` + `ignore_eos=true` 로 고정한다. 길이가 흔들리면 동시성별 비교가 성립하지 않는다
- `IGNORE_EOS=0` 으로 실제 종료 조건 근사, `USE_SCHEMA=1` 로 xgrammar 비용 포함 측정이 가능하다
- 이미지는 합성이다. 실 svnet3 프레임은 반입하지 않으며, 부하 특성은 픽셀 수가 결정한다

## 알려진 한계

- 부하기 cpuset(12-15,44-47)이 서빙 컨테이너 cpuset(0-15,32-47)과 겹친다. 몫 안이라 규칙 위반은 아니지만 측정 오염 요인이므로 `host.csv` 의 `stress_cpu_pct` 로 정도를 확인한다. 완전 분리는 서빙 재기동이 필요해서(dst 영향) 하지 않는다
- 타 팀의 `--cpuset-cpus` 없는 컨테이너가 우리 코어에 스케줄될 수 있다 ([docs/05 도커 사용 시](../docs/05-strad32-team-resource-split.md#도커-사용-시)). `meta.json` 의 컨테이너 스냅샷과 `host.csv` 의 load average 를 함께 봐야 한다
- 합성 native 이미지는 JPEG 압축이 잘 되어 0.27 MB 다. 실 프레임은 수 MB 이므로 HTTP 본문 파싱 비용이 조금 더 크다. 부하기가 서버 안에서 돌아 전송 대역폭은 무관하고, vision 토큰 수는 픽셀 수로 동일하다
- run2 진행 중 r1 이 dst 트래픽을 받으면 DP 비교가 흔들린다. `engine.csv` 의 `success` 증분으로 사후 확인한다
