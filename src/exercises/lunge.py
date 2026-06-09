"""런지: 앞무릎 각도로 렙 카운팅, 깊이/상체 직립 체크.

앞다리(더 많이 굽은 다리)를 자동으로 추적하기 위해 양 무릎 각도 중 더 작은 값을
주 측정값으로 쓴다. 좌우 교대 런지에서도 자연스럽게 동작한다.
"""
from typing import Dict, List, Optional

from ..pose import landmarks as L
from ..pose.angles import joint_angle, segment_angle_to_horizontal
from ..pose.types import Landmarks
from .base import Exercise, FormIssue, visible


class Lunge(Exercise):
    name = "lunge"
    name_ko = "런지"
    primary_metric = "front_knee_angle"
    down_threshold = 120.0   # 앞무릎이 120도 미만으로 굽으면 하강
    up_threshold = 160.0     # 다시 펴지면 렙 완료

    def compute_metrics(self, lm: Landmarks) -> Optional[Dict[str, float]]:
        def p(i):
            return lm[i][:3]

        knees = []
        for hip, knee, ankle in (
            (L.LEFT_HIP, L.LEFT_KNEE, L.LEFT_ANKLE),
            (L.RIGHT_HIP, L.RIGHT_KNEE, L.RIGHT_ANKLE),
        ):
            if visible(lm, hip, knee, ankle):
                knees.append(joint_angle(p(hip), p(knee), p(ankle)))
        if not knees:
            return None

        metrics = {"front_knee_angle": round(min(knees), 1)}
        if len(knees) == 2:
            metrics["rear_knee_angle"] = round(max(knees), 1)

        # 상체 직립도: 어깨-엉덩이 선의 수평 대비 각도(90=수직)
        for shoulder, hip in ((L.LEFT_SHOULDER, L.LEFT_HIP),
                              (L.RIGHT_SHOULDER, L.RIGHT_HIP)):
            if visible(lm, shoulder, hip):
                metrics["torso_angle"] = round(
                    segment_angle_to_horizontal(p(shoulder), p(hip)), 1)
                break
        return metrics

    def check_form(self, metrics: Dict[str, float]) -> List[FormIssue]:
        issues: List[FormIssue] = []
        depth = metrics.get("_depth", metrics["front_knee_angle"])
        if depth > 110:
            issues.append(FormIssue(
                "shallow_depth",
                "앞무릎을 90도까지 더 굽혀 깊이 내려가세요.",
                "major"))
        torso = metrics.get("torso_angle")
        if torso is not None and torso < 65:
            issues.append(FormIssue(
                "torso_lean",
                "상체가 앞으로 숙여졌습니다. 가슴을 펴고 몸통을 세우세요.",
                "minor"))
        return issues
