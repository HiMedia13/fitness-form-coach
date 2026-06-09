"""스쿼트: 무릎 각도로 렙 카운팅, 깊이/등 각도/무릎 모임 체크."""
from typing import Dict, List, Optional

from ..pose import landmarks as L
from ..pose.angles import joint_angle, segment_angle_to_horizontal
from ..pose.types import Landmarks
from .base import Exercise, FormIssue, visible


class Squat(Exercise):
    name = "squat"
    name_ko = "스쿼트"
    primary_metric = "knee_angle"
    down_threshold = 110.0   # 무릎이 110도 미만으로 굽으면 하강
    up_threshold = 160.0     # 무릎이 160도 이상 펴지면 렙 완료

    def compute_metrics(self, lm: Landmarks) -> Optional[Dict[str, float]]:
        if not visible(lm, L.LEFT_HIP, L.LEFT_KNEE, L.LEFT_ANKLE,
                       L.LEFT_SHOULDER):
            # 왼쪽이 안 보이면 오른쪽 시도
            if not visible(lm, L.RIGHT_HIP, L.RIGHT_KNEE, L.RIGHT_ANKLE,
                           L.RIGHT_SHOULDER):
                return None
            hip, knee, ankle = L.RIGHT_HIP, L.RIGHT_KNEE, L.RIGHT_ANKLE
            shoulder = L.RIGHT_SHOULDER
        else:
            hip, knee, ankle = L.LEFT_HIP, L.LEFT_KNEE, L.LEFT_ANKLE
            shoulder = L.LEFT_SHOULDER

        def p(i):
            return lm[i][:3]

        knee_angle = joint_angle(p(hip), p(knee), p(ankle))
        hip_angle = joint_angle(p(shoulder), p(hip), p(knee))
        # 등의 수평 대비 기울기(90=수직). 너무 작으면 상체가 앞으로 숙여짐.
        back_angle = segment_angle_to_horizontal(p(shoulder), p(hip))

        # 무릎 모임(valgus): 양 무릎 간격 / 양 발목 간격 비율
        metrics = {
            "knee_angle": round(knee_angle, 1),
            "hip_angle": round(hip_angle, 1),
            "back_angle": round(back_angle, 1),
        }
        if visible(lm, L.LEFT_KNEE, L.RIGHT_KNEE, L.LEFT_ANKLE, L.RIGHT_ANKLE):
            knee_w = abs(lm[L.LEFT_KNEE][0] - lm[L.RIGHT_KNEE][0])
            ankle_w = abs(lm[L.LEFT_ANKLE][0] - lm[L.RIGHT_ANKLE][0]) + 1e-6
            metrics["knee_ankle_ratio"] = round(knee_w / ankle_w, 2)
        return metrics

    def check_form(self, metrics: Dict[str, float]) -> List[FormIssue]:
        issues: List[FormIssue] = []
        depth = metrics.get("_depth", metrics["knee_angle"])
        if depth > 100:
            issues.append(FormIssue(
                "shallow_depth",
                "스쿼트 깊이가 얕습니다. 허벅지가 수평이 되도록 더 앉으세요.",
                "major"))
        if metrics.get("back_angle", 90) < 35:
            issues.append(FormIssue(
                "torso_lean",
                "상체가 너무 앞으로 숙여졌습니다. 가슴을 펴고 시선을 정면에.",
                "major"))
        ratio = metrics.get("knee_ankle_ratio")
        if ratio is not None and ratio < 0.8:
            issues.append(FormIssue(
                "knee_valgus",
                "무릎이 안쪽으로 모입니다. 무릎을 발끝 방향으로 벌리세요.",
                "major"))
        return issues
