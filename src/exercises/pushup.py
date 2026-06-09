"""푸시업: 팔꿈치 각도로 렙 카운팅, 몸통 일직선/가동범위 체크."""
from typing import Dict, List, Optional

from ..pose import landmarks as L
from ..pose.angles import joint_angle
from ..pose.types import Landmarks
from .base import Exercise, FormIssue, visible


class Pushup(Exercise):
    name = "pushup"
    name_ko = "푸시업"
    primary_metric = "elbow_angle"
    down_threshold = 100.0   # 팔꿈치 100도 미만 → 하강
    up_threshold = 160.0     # 160도 이상 → 렙 완료

    def compute_metrics(self, lm: Landmarks) -> Optional[Dict[str, float]]:
        if visible(lm, L.LEFT_SHOULDER, L.LEFT_ELBOW, L.LEFT_WRIST,
                   L.LEFT_HIP, L.LEFT_ANKLE):
            shoulder, elbow, wrist = L.LEFT_SHOULDER, L.LEFT_ELBOW, L.LEFT_WRIST
            hip, ankle = L.LEFT_HIP, L.LEFT_ANKLE
        elif visible(lm, L.RIGHT_SHOULDER, L.RIGHT_ELBOW, L.RIGHT_WRIST,
                     L.RIGHT_HIP, L.RIGHT_ANKLE):
            shoulder, elbow, wrist = L.RIGHT_SHOULDER, L.RIGHT_ELBOW, L.RIGHT_WRIST
            hip, ankle = L.RIGHT_HIP, L.RIGHT_ANKLE
        else:
            return None

        def p(i):
            return lm[i][:3]

        elbow_angle = joint_angle(p(shoulder), p(elbow), p(wrist))
        # 어깨-엉덩이-발목이 일직선이면 ~180도
        body_line = joint_angle(p(shoulder), p(hip), p(ankle))
        return {
            "elbow_angle": round(elbow_angle, 1),
            "body_line": round(body_line, 1),
        }

    def check_form(self, metrics: Dict[str, float]) -> List[FormIssue]:
        issues: List[FormIssue] = []
        depth = metrics.get("_depth", metrics["elbow_angle"])
        if depth > 95:
            issues.append(FormIssue(
                "shallow_depth",
                "내려가는 깊이가 부족합니다. 가슴이 바닥에 가까워지도록.",
                "major"))
        body = metrics.get("body_line", 180)
        if body < 160:
            issues.append(FormIssue(
                "hip_sag",
                "엉덩이가 처지거나 솟았습니다. 코어에 힘을 줘 몸을 일직선으로.",
                "major"))
        return issues
