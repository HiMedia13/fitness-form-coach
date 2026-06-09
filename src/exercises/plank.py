"""플랭크: 등척성 유지 운동. 몸통 일직선 유지 시간/처짐 체크."""
from typing import Dict, List, Optional

from ..pose import landmarks as L
from ..pose.angles import joint_angle
from ..pose.types import Landmarks
from .base import FormIssue, IsometricExercise, visible


class Plank(IsometricExercise):
    name = "plank"
    name_ko = "플랭크"

    def compute_metrics(self, lm: Landmarks) -> Optional[Dict[str, float]]:
        if visible(lm, L.LEFT_SHOULDER, L.LEFT_HIP, L.LEFT_ANKLE):
            shoulder, hip, ankle = L.LEFT_SHOULDER, L.LEFT_HIP, L.LEFT_ANKLE
        elif visible(lm, L.RIGHT_SHOULDER, L.RIGHT_HIP, L.RIGHT_ANKLE):
            shoulder, hip, ankle = L.RIGHT_SHOULDER, L.RIGHT_HIP, L.RIGHT_ANKLE
        else:
            return None

        def p(i):
            return lm[i][:3]

        body_line = joint_angle(p(shoulder), p(hip), p(ankle))
        return {"body_line": round(body_line, 1)}

    def is_in_position(self, metrics: Dict[str, float]) -> bool:
        # 몸통이 어느 정도 펴져 있으면 플랭크 유지 중으로 간주
        return metrics.get("body_line", 0) > 150

    def check_form(self, metrics: Dict[str, float]) -> List[FormIssue]:
        issues: List[FormIssue] = []
        body = metrics.get("body_line", 180)
        if body < 165:
            issues.append(FormIssue(
                "hip_sag",
                "엉덩이가 처졌습니다. 복부와 둔근에 힘을 줘 일직선을 유지하세요.",
                "major"))
        elif body > 195:
            issues.append(FormIssue(
                "hip_pike",
                "엉덩이가 솟았습니다. 골반을 낮춰 몸을 일직선으로.",
                "minor"))
        return issues
