"""에이전트 루프 오프라인 검증 (API 키 불필요).

1) 실제 영상에서 렙 궤적을 캡처하고 도구(dispatch)가 올바른 데이터를 주는지 확인.
2) 가짜 anthropic 클라이언트로 AgentCoach 의 도구 사용 루프를 구동:
   list_reps → get_rep_trajectory → submit_coaching 흐름이 Coaching 을 반환하는지 검증.

사용:
    python tools/test_agent_offline.py <영상경로> [운동명]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from src.coach.agent import AgentCoach  # noqa: E402
from src.coach.tools import dispatch  # noqa: E402
from src.exercises import registry  # noqa: E402
from src.exercises.base import RepResult  # noqa: E402
from src.pose.estimator import PoseEstimator  # noqa: E402


def capture_reps(video, ex_name):
    exercise = registry.create(ex_name)
    est = PoseEstimator()
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    reps, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = (cap.get(cv2.CAP_PROP_POS_MSEC) or 0) / 1000.0 or idx / fps
        idx += 1
        lm = est.process(frame)
        if lm is not None:
            r = exercise.update(lm, t)
            if isinstance(r, RepResult):
                reps.append(r)
    cap.release()
    est.close()
    return exercise.name, reps


# --- 가짜 anthropic 클라이언트: 미리 정해진 도구 호출 시퀀스를 재생 ---
class _Block:
    def __init__(self, type, name=None, input=None, id=None):
        self.type, self.name, self.input, self.id = type, name, input, id


class _Resp:
    def __init__(self, content, stop_reason):
        self.content, self.stop_reason = content, stop_reason


class _Messages:
    def __init__(self):
        self.turn = 0
        self.calls = []

    def create(self, **kw):
        self.turn += 1
        if self.turn == 1:
            return _Resp([_Block("tool_use", "list_reps", {}, "t1")], "tool_use")
        if self.turn == 2:
            return _Resp([_Block("tool_use", "get_rep_trajectory",
                                 {"rep_index": 1, "metric": "knee_angle"}, "t2")],
                         "tool_use")
        # 마지막: 최종 코칭 제출
        return _Resp([_Block("tool_use", "submit_coaching", {
            "form_score": 82, "severity": "minor",
            "headline": "깊이는 좋아요",
            "cues": ["후반 렙에서 깊이를 유지하세요"],
            "encouragement": "좋은 세트였습니다!",
        }, "t3")], "tool_use")


class _FakeClient:
    def __init__(self):
        self.messages = _Messages()


def main():
    video = sys.argv[1]
    ex_name = sys.argv[2] if len(sys.argv) > 2 else "squat"
    name, reps = capture_reps(video, ex_name)
    print(f"캡처: 운동={name}, 렙={len(reps)}")
    if not reps:
        print("렙 미감지 — 종료")
        return
    print(f"렙1 궤적 프레임 수: {len(reps[0].trajectory)}")

    rep_map = {r.rep_index: r for r in reps}
    print("\n[도구] list_reps:")
    print("  " + dispatch("list_reps", {}, rep_map))
    print("\n[도구] get_rep_stats(1):")
    print("  " + dispatch("get_rep_stats", {"rep_index": 1}, rep_map))
    print("\n[도구] get_rep_trajectory(1, knee_angle):")
    print("  " + dispatch("get_rep_trajectory",
                          {"rep_index": 1, "metric": "knee_angle"}, rep_map))

    # 에이전트 루프 (가짜 클라이언트)
    print("\n[에이전트 루프 — 가짜 클라이언트]")
    agent = object.__new__(AgentCoach)
    agent.model = "fake"
    agent.system = [{"type": "text", "text": "sys"}]
    agent.client = _FakeClient()
    summary = {"exercise": name, "event": "set", "total_reps": len(reps),
               "_records": reps}
    result = agent.coach(summary)
    print("  반환 타입:", type(result).__name__)
    print("  코칭:", result.form_score, result.severity, "|",
          result.headline, "|", result.cues)
    print("  API 호출 턴 수:", agent.client.messages.turn)
    print("\nAGENT LOOP OK")


if __name__ == "__main__":
    main()
