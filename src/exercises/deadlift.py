"""데드리프트: 엉덩이 힌지 각도로 렙 카운팅, 등 중립/가동범위 체크."""
from typing import Dict, List, Optional

from ..pose import landmarks as L
from ..pose.angles import joint_angle, segment_angle_to_horizontal
from ..pose.types import Landmarks
from .base import Exercise, FormIssue, visible


class Deadlift(Exercise):
    name = "deadlift"
    name_ko = "데드리프트"
    primary_metric = "hip_angle"
    down_threshold = 120.0   # 엉덩이 힌지로 상체가 숙여지면 hip_angle 감소
    up_threshold = 165.0     # 락아웃(직립) 시 hip_angle 증가

    def compute_metrics(self, lm: Landmarks) -> Optional[Dict[str, float]]:
        if visible(lm, L.LEFT_SHOULDER, L.LEFT_HIP, L.LEFT_KNEE, L.LEFT_ANKLE):
            shoulder, hip, knee, ankle = (
                L.LEFT_SHOULDER, L.LEFT_HIP, L.LEFT_KNEE, L.LEFT_ANKLE)
        elif visible(lm, L.RIGHT_SHOULDER, L.RIGHT_HIP, L.RIGHT_KNEE, L.RIGHT_ANKLE):
            shoulder, hip, knee, ankle = (
                L.RIGHT_SHOULDER, L.RIGHT_HIP, L.RIGHT_KNEE, L.RIGHT_ANKLE)
        else:
            return None

        def p(i):
            return lm[i][:3]

        hip_angle = joint_angle(p(shoulder), p(hip), p(knee))
        knee_angle = joint_angle(p(hip), p(knee), p(ankle))
        back_angle = segment_angle_to_horizontal(p(shoulder), p(hip))
        return {
            "hip_angle": round(hip_angle, 1),
            "knee_angle": round(knee_angle, 1),
            "back_angle": round(back_angle, 1),
        }

    def check_form(self, metrics: Dict[str, float]) -> List[FormIssue]:
        issues: List[FormIssue] = []
        # 락아웃에서 충분히 서지 못함
        if metrics.get("hip_angle", 180) < 160:
            issues.append(FormIssue(
                "incomplete_lockout",
                "마무리에서 엉덩이를 끝까지 펴 직립하세요.",
                "minor"))
        # 바닥 구간에서 등이 너무 수평(말림 위험) — 무릎이 거의 안 굽었는데 상체만 숙임
        if metrics.get("back_angle", 90) < 20 and metrics.get("knee_angle", 0) > 150:
            issues.append(FormIssue(
                "rounded_back_risk",
                "무릎은 펴진 채 등만 숙여졌습니다. 무릎을 굽혀 엉덩이로 힌지하고 등을 중립으로.",
                "major"))
        return issues
