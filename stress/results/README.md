# 부하 테스트 원자료

2026-07-30 실측. 해석과 결론은 [docs/09](../../docs/09-stress-test-results.md) 에 있다.
이 디렉토리는 그 근거가 되는 원자료이며, 수치를 다시 검산하거나 다른 각도로 볼 때 쓴다.

서버 원본 위치는 `/data01/dmt/stress-results/` 이고, Locust 컨테이너를 내리면
`export/` 의 내용은 서버에서 사라진다(UI 통계가 메모리에만 있기 때문). 그래서 레포에 담았다.

## 런 구성

| 디렉토리 | 대상 | 동시성 단계 | 시나리오 | 목적 |
|---|---|---|---|---|
| `*-run1` | LB `:8000` | 1,2,4,8,12,16,24,32,48,64 | 혼합 | 주 결과. 포화점, 지연 곡선, 파괴점 |
| `*-run2` | 레플리카 `:8001` 직결 | 1,2,4,8,12,16,24,32 | 혼합 | DP=2 확장 효율 비교 |
| `*-run3a` | LB `:8000` | 8,16 | S1 텍스트만 | 해상도별 비용 |
| `*-run3b` | LB `:8000` | 8,16 | S2 1MP | 해상도별 비용 |
| `*-run3c` | LB `:8000` | 8,16 | S3 native 4464x2160 | 해상도별 비용 |
| `*-knee` | LB `:8000` | 8,12,16,24,32 (단계 180초) | 혼합 | 포화점 재현성 확인. 90초 단계의 변동 검증 |
| `*-seqs32` | 실험 레플리카 `:8003` | 1,2,4,8,12,16,24,32 | 혼합 | `--max-num-seqs` 32 A/B (run2 와 동일 조건) |
| `*-dp4` | LB `:8000` (레플리카 4대) | 8,16,24,32,48,64,96 (단계 180초) | 혼합 | DP=4 확장 후 재측정 |
| `*-smoke` | LB `:8000` | 1,4 | 혼합 | 하네스 검증용. 본 결과에 쓰지 않음 |

단계당 90초(`*-knee`, `*-dp4` 는 180초)이며 앞 30초는 워밍으로 버리고 나머지를 측정 구간으로 쓴다.
혼합 비중은 텍스트 2, 1MP 5, native 3 이다.

## 파일

| 파일 | 생성자 | 내용 |
|---|---|---|
| `summary.md` | `analyze.py` | 동시성별 표, 판정, 무침범 증빙 |
| `meta.json` | `run.sh` | 실행 구성, vLLM 플래그 스냅샷, 시작 시점 컨테이너와 GPU 상태 |
| `locust_stats_history.csv` | Locust | 클라이언트 측 시계열 원자료. `analyze.py` 의 주 입력 |
| `locust_stats.csv` | Locust | 전체 기간 집계 |
| `locust_failures.csv`, `locust_exceptions.csv` | Locust | 실패와 예외 (전 런 0건이라 헤더만) |
| `engine.csv` | `monitor.sh` | 레플리카별 엔진 지표 1초 샘플 (running, waiting, KV, preemption, 생성 토큰). `*-seqs32` 는 감시 대상 포트 버그로 비어 있다 (docs/09 8번) |
| `gpu.csv` | `monitor.sh` | GPU 8장의 util, mem, power, temp, SM 클럭 1초 샘플 |
| `host.csv` | `monitor.sh` | load average, MemAvailable, 컨테이너 메모리, NUMA free, LB 5xx, 컨테이너 CPU |
| `guard.log` | `monitor.sh` | 가드 판정 이력. 타 팀 GPU 클럭 기준선 포함 |
| `export/` | `export.sh` | Locust UI 스냅샷 (아래) |

`export/` 는 UI 탭 구성과 같다.

| 파일 | UI 대응 |
|---|---|
| `statistics.csv` | Statistics. 시나리오별 요청 수와 지연 분위. `Average Content Size` 가 TTFT 행은 프롬프트 토큰, total 행은 출력 토큰이다 |
| `charts_report.html` | Charts. RPS, 응답시간, 사용자 수 시계열이 내장되어 있다. 브라우저로 바로 열면 된다 |
| `failures.csv`, `exceptions.csv` | Failures, Exceptions |
| `ratio.json` | Current ratio. 시나리오 가중치 실측 |
| `locust_logs.json` | Logs |
| `stats_snapshot.json` | 종료 시점 상태 전체 |
| `container.log` | 컨테이너 표준 출력. warmup 결과와 shape 전환 이력 |

## 다시 집계하기

`analyze.py` 는 표준 라이브러리만 쓰므로 어디서든 돌아간다. 판정 규칙을 바꿔 보거나
측정 창을 조정할 때 원자료에서 다시 뽑으면 된다.

```bash
python3 stress/analyze.py stress/results/20260730T025229Z-run1
```

## 이 측정의 조건

- 서빙 구성은 `meta.json` 의 `vllm_args` 에 그대로 남아 있다 (`--max-num-seqs 16`,
  `--gpu-memory-utilization 0.95`, `--max-model-len 32768` 등)
- 측정 중 타 팀 컨테이너가 돌고 있었다. `meta.json` 의 `containers_at_start` 와
  `host.csv` 의 load average 로 확인할 수 있다
- 이미지는 합성이다. vision 토큰 수는 픽셀 수로 결정되므로 부하 특성은 실 프레임과 같다
