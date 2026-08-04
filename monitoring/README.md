# 관측 스택 (Prometheus + Grafana)

[docs/10 3번](../docs/10-operations-metrics-and-slo.md#3-상시-관측-최소-세트)의 상시 관측 구현.
자작 코드 없음: 기성 컨테이너 2개 + 이 디렉토리의 설정 파일이 전부다.

- 가동: 2026-08-04, strad32. `vlm-prom`(:9090) + `vlm-grafana`(:3000)
- 접속: `http://${STRAD32_IP}:3000` (admin 비밀번호는 기동 시 환경변수로 주입, 기록하지 않음)
- 스크레이프: 운영 레플리카 4대(:8001-8004) 엔진 `/metrics`, 15초 간격, 보존 15일
- dcgm-exporter(GPU 지표)는 2차 보류 ([docs/04 5번](../docs/04-gpu-pinning-and-serving.md#5-모니터링)).
  엔진 지표가 판단 규칙의 1순위라서다

## 구성 파일

| 파일 | 역할 |
|---|---|
| `prometheus.yml` | 스크레이프 대상 (레플리카 증감 시 여기 수정 후 vlm-prom 재기동) |
| `provisioning/datasources/prometheus.yml` | Grafana 데이터소스 (uid `prometheus` 고정) |
| `provisioning/dashboards/vllm.yml` | 대시보드 자동 로드 설정 |
| `dashboards/*.json` | vLLM 공식 대시보드 3종 (upstream `examples/observability/`에서 가져와 `${DS_PROMETHEUS}` -> `prometheus` 치환. 경로가 자주 바뀌어 vendor함) |

## 기동 (strad32, dmt 계정)

데이터/설정 위치: `/data01/dmt/monitoring/` (이 디렉토리 내용 + `prometheus-data/`, `grafana-data/`)

```bash
docker run -d --name vlm-prom --network host --restart unless-stopped \
  --cpuset-cpus 0-15,32-47 --cpuset-mems 0 --memory 1g \
  -v /data01/dmt/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro \
  -v /data01/dmt/monitoring/prometheus-data:/prometheus \
  prom/prometheus:latest \
  --config.file=/etc/prometheus/prometheus.yml \
  --storage.tsdb.path=/prometheus --storage.tsdb.retention.time=15d

docker run -d --name vlm-grafana --network host --restart unless-stopped \
  --cpuset-cpus 0-15,32-47 --cpuset-mems 0 --memory 1g \
  -e GF_SECURITY_ADMIN_PASSWORD='<기동자가 정함>' \
  -e GF_ANALYTICS_REPORTING_ENABLED=false \
  -v /data01/dmt/monitoring/provisioning:/etc/grafana/provisioning:ro \
  -v /data01/dmt/monitoring/dashboards:/var/lib/grafana/dashboards:ro \
  -v /data01/dmt/monitoring/grafana-data:/var/lib/grafana \
  grafana/grafana-oss:latest
```

- cpuset/RAM 상한은 dst+dmt 몫 규칙([docs/05](../docs/05-strad32-team-resource-split.md)) 준수 (각 1g, 합 2g)
- 포트 3000/9090은 호스트 직결(host network). 타 팀 포트와 충돌 없음 확인 후 기동함

## 검증

```bash
curl -s localhost:9090/api/v1/targets   # 4 타겟 전부 "health": "up"
curl -s localhost:3000/api/health       # database ok
```
