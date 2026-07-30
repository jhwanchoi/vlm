"""부하의 형태(시간에 따른 동시성)만 담당한다. 요청 내용은 모른다.

STAGES 의 각 값을 STAGE_SEC 초씩 유지하는 계단이다. 마지막 단계가 끝나면 tick 이
None 을 반환해 테스트가 스스로 멈춘다 (무한 부하 방지의 1차 장치. 2차는 run.sh 의
--run-time, 3차는 monitor.sh 의 가드).

analyze.py 는 각 단계의 앞 WARMUP_SEC 초를 버리고 나머지를 측정 구간으로 쓴다.
단계 경계에서 사용자가 스폰되는 동안의 과도 구간을 수치에서 빼기 위함이다.
"""
import os

from locust import LoadTestShape

STAGES = [int(x) for x in os.environ.get("STAGES", "1,2,4,8,12,16,24,32,48,64").split(",")]
STAGE_SEC = int(os.environ.get("STAGE_SEC", "90"))
WARMUP_SEC = int(os.environ.get("WARMUP_SEC", "30"))


class StepLoad(LoadTestShape):
    def tick(self):
        idx = int(self.get_run_time() // STAGE_SEC)
        if idx >= len(STAGES):
            return None
        users = STAGES[idx]
        # spawn_rate 를 사용자 수와 같게 두어 단계 전환이 1초 안에 끝나게 한다.
        return users, max(1, users)
