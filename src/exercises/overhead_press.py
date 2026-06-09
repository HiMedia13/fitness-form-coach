"""오버헤드 프레스: 팔꿈치 각도로 렙 카운팅, 락아웃/오버헤드 도달 체크.

랙 위치(팔꿈치 ~90도)에서 머리 위 락아웃(~170도)으로 폈다가 돌아오는 동작을
한 렙으로 센다. 주 측정값은 팔꿈치 각도.
"""
from typing import Dict, List, Optional

from ..pose import landmarks as L
from ..pose.angles import joint_angle
from ..pose.types import Landmarks
from .base import Exercise, FormIssue, visible


class OverheadPress(Exercise):
    name = "overhead_press"
    name_ko = "오버헤드 프레스"
    primary_metric = "elbow_angle"
    down_threshold = 120.0   # 랙 위치(팔꿈치 굽힘)
    up_threshold = 160.0     # 머리 위 락아웃(팔꿈치 폄)

    def compute_metrics(self, lm: Landmarks) -> Optional[Dict[str, float]]:
        if visible(lm, L.LEFT_SHOULDER, L.LEFT_ELBOW, L.LEFT_WRIST):
            shoulder, elbow, wrist = L.LEFT_SHOULDER, L.LEFT_ELBOW, L.LEFT_WRIST
        elif visible(lm, L.RIGHT_SHOULDER, L.RIGHT_ELBOW, L.RIGHT_WRIST):
            shoulder, elbow, wrist = L.RIGHT_SHOULDER, L.RIGHT_ELBOW, L.RIGHT_WRIST
        else:
            return None

        def p(i):
            return lm[i][:3]

        elbow_angle = joint_angle(p(shoulder), p(elbow), p(wrist))
        # 이미지 y는 아래로 증가 → 손목 y가 어깨 y보다 작으면 손목이 더 높음
        wrist_above_shoulder = 1.0 if lm[wrist][1] < lm[shoulder][1] else 0.0
        return {
            "elbow_angle": round(elbow_angle, 1),
            "wrist_above_shoulder": wrist_above_shoulder,
        }

    def check_form(self, metrics: Dict[str, float]) -> List[FormIssue]:
        issues: List[FormIssue] = []
        # 랙 깊이: 사이클 최저 팔꿈치 각도가 너무 크면 충분히 안 내림(가동범위 부족)
        depth = metrics.get("_depth", metrics["elbow_angle"])
        if depth > 110:
            issues.append(FormIssue(
                "partial_rom",
                "시작에서 팔꿈치를 90도 어깨 높이까지 충분히 내리세요.",
                "minor"))
        # 락아웃에서 손목이 어깨 위로 올라가지 않으면 머리 위로 끝까지 못 뻗은 것
        if metrics.get("wrist_above_shoulder", 1.0) < 1.0:
            issues.append(FormIssue(
                "incomplete_overhead",
                "팔을 머리 위로 끝까지 뻗어 락아웃하세요.",
                "major"))
        return issues
