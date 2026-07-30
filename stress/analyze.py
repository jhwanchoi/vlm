#!/usr/bin/env python3
"""부하 테스트 결과 집계. 표준 라이브러리만 사용한다 (서버에 pip 이 없다).

    python3 analyze.py <결과디렉토리> > summary.md

입력
  locust_stats_history.csv   클라이언트 측 RPS, 지연 분포, 실패 (Locust --csv-full-history)
  engine.csv                 레플리카별 엔진 지표 1초 샘플 (monitor.sh)
  gpu.csv, host.csv          시스템 지표 1초 샘플 (monitor.sh)
  meta.json                  실행 구성

처리량은 클라이언트 카운트가 아니라 엔진의 generation_tokens_total 증분으로 낸다.
Locust 의 Average Content Size 는 테스트 시작부터의 누적 평균이라 단계별 값으로
쓸 수 없다.
"""
import csv
import json
import pathlib
import statistics
import sys


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def fnum(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def ts_to_epoch(s):
    """monitor.sh 의 UTC ISO 문자열을 epoch 초로 바꾼다."""
    import calendar
    import time as _t

    return calendar.timegm(_t.strptime(s, "%Y-%m-%dT%H:%M:%SZ"))


def stages_from_history(rows, warmup):
    """user count 가 유지되는 구간을 단계로 묶고 측정 창(워밍 이후)을 계산한다."""
    seen = []
    for r in rows:
        users = int(fnum(r.get("User Count")))
        ts = int(fnum(r.get("Timestamp")))
        if not seen or seen[-1]["users"] != users:
            seen.append({"users": users, "start": ts, "end": ts})
        else:
            seen[-1]["end"] = ts
    for s in seen:
        # 단계가 워밍 구간보다 짧으면(짧은 스모크 등) 뒤쪽 1/3 을 측정 창으로 쓴다.
        dur = s["end"] - s["start"]
        s["win_start"] = s["start"] + (warmup if dur > warmup + 10 else dur // 3)
        s["win_end"] = s["end"]
    return [s for s in seen if s["users"] > 0 and s["win_end"] > s["win_start"]]


def per_name(rows, stage, req_type):
    out = {}
    for r in rows:
        if r.get("Type") != req_type:
            continue
        ts = int(fnum(r.get("Timestamp")))
        if not (stage["win_start"] <= ts <= stage["win_end"]):
            continue
        out.setdefault(r.get("Name", "?"), []).append(r)
    return out


def agg_stage(rows, stage):
    """단계별 클라이언트 지표."""
    totals = per_name(rows, stage, "total")
    ttfts = per_name(rows, stage, "TTFT")

    rps = 0.0
    p95s, names = [], {}
    req_delta = fail_delta = 0
    for name, rs in totals.items():
        rps += statistics.fmean(fnum(r.get("Requests/s")) for r in rs)
        p95 = [fnum(r.get("95%")) for r in rs if fnum(r.get("95%")) > 0]
        p50 = [fnum(r.get("50%")) for r in rs if fnum(r.get("50%")) > 0]
        if p95:
            p95s.append(max(p95))
        rc = [fnum(r.get("Total Request Count")) for r in rs]
        fc = [fnum(r.get("Total Failure Count")) for r in rs]
        if rc:
            req_delta += max(rc) - min(rc)
        if fc:
            fail_delta += max(fc) - min(fc)
        names[name] = {
            "p50": statistics.fmean(p50) if p50 else 0.0,
            "p95": max(p95) if p95 else 0.0,
        }

    ttft = {}
    for name, rs in ttfts.items():
        p50 = [fnum(r.get("50%")) for r in rs if fnum(r.get("50%")) > 0]
        p95 = [fnum(r.get("95%")) for r in rs if fnum(r.get("95%")) > 0]
        ttft[name] = {
            "p50": statistics.fmean(p50) if p50 else 0.0,
            "p95": max(p95) if p95 else 0.0,
        }

    return {
        "rps": rps,
        "p95": max(p95s) if p95s else 0.0,
        "requests": req_delta,
        "failures": fail_delta,
        "fail_pct": (fail_delta / req_delta * 100) if req_delta else 0.0,
        "by_name": names,
        "ttft": ttft,
    }


def engine_stage(rows, stage, ports):
    """단계별 엔진 지표. 처리량은 generation_tokens_total 증분에서 낸다."""
    out = {"tok_s": 0.0, "running": 0.0, "waiting": 0.0, "kv": 0.0, "preempt": 0.0, "success": 0.0}
    span = max(1, stage["win_end"] - stage["win_start"])
    for port in ports:
        sel = [
            r for r in rows
            if r.get("port") == str(port) and stage["win_start"] <= ts_to_epoch(r["ts"]) <= stage["win_end"]
        ]
        if not sel:
            continue
        gen = [fnum(r.get("gen_tokens")) for r in sel]
        pre = [fnum(r.get("preemptions")) for r in sel]
        suc = [fnum(r.get("success")) for r in sel]
        out["tok_s"] += (max(gen) - min(gen)) / span
        out["preempt"] += max(pre) - min(pre)
        out["success"] += max(suc) - min(suc)
        out["running"] += statistics.fmean(fnum(r.get("running")) for r in sel)
        out["waiting"] += statistics.fmean(fnum(r.get("waiting")) for r in sel)
        out["kv"] = max(out["kv"], max(fnum(r.get("kv_usage")) for r in sel))
    return out


def system_stage(gpu_rows, host_rows, stage, ours, others):
    def sel(rows):
        return [r for r in rows if stage["win_start"] <= ts_to_epoch(r["ts"]) <= stage["win_end"]]

    g = sel(gpu_rows)
    h = sel(host_rows)
    ours_r = [r for r in g if r.get("gpu") in {str(x) for x in ours}]
    other_r = [r for r in g if r.get("gpu") in {str(x) for x in others}]
    return {
        "our_temp_max": max((fnum(r.get("temp_c")) for r in ours_r), default=0.0),
        "our_power_max": max((fnum(r.get("power_w")) for r in ours_r), default=0.0),
        "our_util_mean": statistics.fmean([fnum(r.get("util")) for r in ours_r]) if ours_r else 0.0,
        "other_temp_max": max((fnum(r.get("temp_c")) for r in other_r), default=0.0),
        "other_sm_min": min((fnum(r.get("sm_mhz")) for r in other_r), default=0.0),
        "load_max": max((fnum(r.get("load1")) for r in h), default=0.0),
        "mem_avail_min": min((fnum(r.get("mem_avail_mb")) for r in h if r.get("mem_avail_mb")), default=0.0),
        "our_mem_max": max((fnum(r.get("our_mem_gib")) for r in h if r.get("our_mem_gib")), default=0.0),
        "node0_free_min": min((fnum(r.get("node0_free_mb")) for r in h if r.get("node0_free_mb")), default=0.0),
        "stress_cpu_max": max((fnum(r.get("stress_cpu_pct")) for r in h if r.get("stress_cpu_pct")), default=0.0),
    }


def verdicts(table):
    knee = knee_note = None
    for prev, cur in zip(table, table[1:]):
        if prev["eng"]["tok_s"] <= 0:
            continue
        growth = (cur["eng"]["tok_s"] - prev["eng"]["tok_s"]) / prev["eng"]["tok_s"] * 100
        lat = (cur["cli"]["p95"] / prev["cli"]["p95"]) if prev["cli"]["p95"] else 0
        if growth < 10 and lat > 1.5:
            knee, knee_note = prev["users"], f"처리량 증가 {growth:.1f}%, p95 {lat:.2f}배"
            break
        # 지연만 늘어난 것은 포화가 아니다. 폐루프 부하에서는 처리량이 크게 늘 때도
        # 지연이 함께 늘어난다. 처리량 증가가 실제로 둔화된 경우만 후보로 본다.
        if knee is None and growth < 20:
            knee, knee_note = prev["users"], f"후보 (처리량 증가 {growth:.1f}%, p95 {lat:.2f}배)"
    break_pt = next((r["users"] for r in table if r["cli"]["fail_pct"] > 1.0), None)
    preempt_pt = next((r["users"] for r in table if r["eng"]["preempt"] > 0), None)
    return knee, knee_note, break_pt, preempt_pt


def main():
    rdir = pathlib.Path(sys.argv[1])
    meta = {}
    if (rdir / "meta.json").exists():
        try:
            meta = json.loads((rdir / "meta.json").read_text())
        except json.JSONDecodeError:
            meta = {}
    warmup = int(meta.get("warmup_sec", 30))
    target = str(meta.get("target", ""))
    ports = [8001, 8002] if target.endswith(":8000") else [int(target.rsplit(":", 1)[-1] or 8001)]

    hist = read_csv(rdir / "locust_stats_history.csv")
    engine = read_csv(rdir / "engine.csv")
    gpu = read_csv(rdir / "gpu.csv")
    host = read_csv(rdir / "host.csv")
    if not hist:
        print("locust_stats_history.csv 가 없다. 집계할 수 없다.")
        return 1

    stages = stages_from_history(hist, warmup)
    if not stages:
        print("# 집계 불가\n\n측정 창이 없다. 테스트가 너무 짧았거나 즉시 중단됐다.")
        if (rdir / "guard_triggered.txt").exists():
            print(f"\n가드 발동: {(rdir / 'guard_triggered.txt').read_text().strip()}")
        return 0
    table = []
    for st in stages:
        table.append({
            "users": st["users"],
            "sec": st["win_end"] - st["win_start"],
            "cli": agg_stage(hist, st),
            "eng": engine_stage(engine, st, ports),
            "sys": system_stage(gpu, host, st, [0, 1], [4, 5, 6, 7]),
        })

    knee, knee_note, break_pt, preempt_pt = verdicts(table)

    out = []
    out.append(f"# 스트레스 테스트 결과: {meta.get('name', rdir.name)}\n")
    out.append(f"- 대상: `{target}` (레플리카 {ports})")
    out.append(f"- 구성: 단계 `{meta.get('stages')}` × {meta.get('stage_sec')}s, 시나리오 `{meta.get('scenarios')}`, "
               f"max_tokens {meta.get('max_tokens')}, ignore_eos {meta.get('ignore_eos')}, use_schema {meta.get('use_schema')}")
    out.append(f"- 시작: {meta.get('started_utc')}  결과: `{rdir}`")
    if (rdir / "guard_triggered.txt").exists():
        out.append(f"- **가드 발동**: {(rdir / 'guard_triggered.txt').read_text().strip()}")
    out.append("")

    out.append("## 동시성별 실측\n")
    out.append("| 동시 | RPS | 출력 tok/s | TTFT p50 (ms) | TTFT p95 (ms) | 전체 p95 (ms) | 실패% | running | waiting | KV% | preempt | 우리GPU 평균util |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in table:
        t50 = statistics.fmean([v["p50"] for v in r["cli"]["ttft"].values()]) if r["cli"]["ttft"] else 0
        t95 = max([v["p95"] for v in r["cli"]["ttft"].values()], default=0)
        out.append(
            f"| {r['users']} | {r['cli']['rps']:.2f} | {r['eng']['tok_s']:.0f} | {t50:.0f} | {t95:.0f} | "
            f"{r['cli']['p95']:.0f} | {r['cli']['fail_pct']:.2f} | {r['eng']['running']:.1f} | "
            f"{r['eng']['waiting']:.1f} | {r['eng']['kv'] * 100:.1f} | {r['eng']['preempt']:.0f} | "
            f"{r['sys']['our_util_mean']:.0f} |"
        )
    out.append("")

    out.append("## 시나리오별 지연 (마지막 단계)\n")
    if table:
        last = table[-1]
        out.append("| 시나리오 | TTFT p50 (ms) | TTFT p95 (ms) | 전체 p50 (ms) | 전체 p95 (ms) |")
        out.append("|---|---|---|---|---|")
        for name in sorted(set(last["cli"]["by_name"]) | set(last["cli"]["ttft"])):
            tt = last["cli"]["ttft"].get(name, {})
            tot = last["cli"]["by_name"].get(name, {})
            out.append(f"| {name} | {tt.get('p50', 0):.0f} | {tt.get('p95', 0):.0f} | "
                       f"{tot.get('p50', 0):.0f} | {tot.get('p95', 0):.0f} |")
    out.append("")

    out.append("## 판정\n")
    out.append(f"- 포화점(knee): **{knee if knee else '미검출'}** {f'({knee_note})' if knee_note else ''}")
    out.append(f"- 파괴점(실패율 1% 초과): **{break_pt if break_pt else '미검출'}**")
    out.append(f"- 실질 상한(preemption 시작): **{preempt_pt if preempt_pt else '미발생'}**")
    limits = [x for x in (knee, break_pt, preempt_pt) if x]
    if limits:
        base = min(limits)
        out.append(f"- **권고 동시성: {max(1, int(base * 0.7))}** (기준 {base} 의 70%)")
    peak = max((r["eng"]["tok_s"] for r in table), default=0)
    out.append(f"- 최대 출력 처리량: **{peak:.0f} tok/s**")
    out.append("")

    out.append("## 타 팀 무침범 증빙\n")
    out.append("| 항목 | 최대/최소 |")
    out.append("|---|---|")
    out.append(f"| 타 팀 GPU(4-7) 최고 온도 | {max((r['sys']['other_temp_max'] for r in table), default=0):.0f} 도 |")
    out.append(f"| 타 팀 GPU(4-7) 최저 SM 클럭 | {min((r['sys']['other_sm_min'] for r in table if r['sys']['other_sm_min']), default=0):.0f} MHz |")
    out.append(f"| 우리 GPU(0-1) 최고 온도 | {max((r['sys']['our_temp_max'] for r in table), default=0):.0f} 도 |")
    out.append(f"| 우리 GPU(0-1) 최고 전력 | {max((r['sys']['our_power_max'] for r in table), default=0):.0f} W |")
    out.append(f"| load average 최대 | {max((r['sys']['load_max'] for r in table), default=0):.1f} |")
    out.append(f"| MemAvailable 최소 | {min((r['sys']['mem_avail_min'] for r in table if r['sys']['mem_avail_min']), default=0):.0f} MB |")
    out.append(f"| 우리 컨테이너 메모리 최대 | {max((r['sys']['our_mem_max'] for r in table), default=0):.1f} GiB (몫 225g) |")
    out.append(f"| NUMA node0 최소 free (참고, page cache 제외) | {min((r['sys']['node0_free_min'] for r in table if r['sys']['node0_free_min']), default=0):.0f} MB |")
    out.append(f"| 부하기 컨테이너 CPU 최대 | {max((r['sys']['stress_cpu_max'] for r in table), default=0):.0f} % |")
    out.append("")
    out.append("부하기는 GPU 를 요청하지 않으며 cpuset 은 dst+dmt 배정의 부분집합이다. "
               "타 팀 GPU 온도와 클럭은 감시 대상이며, 임계 초과 시 자동 중단된다.")

    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
