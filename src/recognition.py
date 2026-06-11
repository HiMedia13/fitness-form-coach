"""운동 자동 인식기 (휴리스틱).

세트 동안 매 프레임 포즈 특징을 누적하고, 세트 종료 시 어떤 운동이었는지 추정한다.
ML 모델 없이 구별력 있는 포즈 특징으로 점수를 매긴다:

- 몸통 방향: 수평이면 푸시업/플랭크, 수직이면 스쿼트/데드리프트/런지/오버헤드프레스
- 팔꿈치 가동범위 큼 → 푸시업/오버헤드프레스
- 손목이 머리 위로 자주 올라감 → 오버헤드프레스
- 무릎 가동범위 큼 → 스쿼트/런지(/데드리프트)
- 좌우 무릎 비대칭 큼 → 런지(스태거드 스탠스)
- 몸통 각도 변화(힌지) 큼 → 데드리프트

휴리스틱이라 완벽하지 않다. 임계값은 src 상단 상수로 조정 가능.
"""
from typing import Dict, Optional, Tuple

from .pose import landmarks as L
from .pose.angles import joint_angle, midpoint, segment_angle_to_horizontal
from .pose.types import Landmarks

_VIS = 0.5
MIN_FRAMES = 15                 # 이보다 적게 관측되면 판단 보류
AUTO_SWITCH_CONFIDENCE = 0.55   # 1등이 2등보다 충분히 우세 + 현재와 다르면 자동 전환


def _vis(lm: Landmarks, *idxs: int) -> bool:
    return all(lm.get(i, (0, 0, 0, 0))[3] >= _VIS for i in idxs)


def _side(lm: Landmarks, left, right):
    """왼쪽이 보이면 왼쪽, 아니면 오른쪽 인덱스 튜플을 반환. 둘 다 없으면 None."""
    if _vis(lm, *left):
        return left
    if _vis(lm, *right):
        return right
    return None


class ExerciseRecognizer:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._n = 0
        self._torso_sum = 0.0
        self._torso_min = float("inf")
        self._torso_max = float("-inf")
        self._elbow_min = float("inf")
        self._elbow_max = float("-inf")
        self._elbow_n = 0
        self._knee_min = float("inf")
        self._knee_max = float("-inf")
        self._knee_n = 0
        self._knee_asym_sum = 0.0     # 좌우 무릎 각도차 누적(평균용)
        self._knee_asym_n = 0
        self._ankle_stag_sum = 0.0    # 좌우 발목 수직(y) 간격 누적 = 앞뒤 스태거
        self._ankle_stag_n = 0
        self._wrist_oh_n = 0
        self._wrist_n = 0
        self._hip_y_min = float("inf")
        self._hip_y_max = float("-inf")

    def feed(self, lm: Landmarks) -> None:
        def p(i):
            return lm[i][:3]

        # 몸통 방향 (어깨-엉덩이). 양쪽 가능하면 중점 사용.
        if _vis(lm, L.LEFT_SHOULDER, L.LEFT_HIP, L.RIGHT_SHOULDER, L.RIGHT_HIP):
            sh = midpoint(p(L.LEFT_SHOULDER), p(L.RIGHT_SHOULDER))
            hp = midpoint(p(L.LEFT_HIP), p(L.RIGHT_HIP))
        else:
            sd = _side(lm, (L.LEFT_SHOULDER, L.LEFT_HIP),
                       (L.RIGHT_SHOULDER, L.RIGHT_HIP))
            if sd is None:
                return
            sh, hp = p(sd[0]), p(sd[1])

        self._n += 1
        torso = segment_angle_to_horizontal(sh, hp)  # 90=수직, 0=수평
        self._torso_sum += torso
        self._torso_min = min(self._torso_min, torso)
        self._torso_max = max(self._torso_max, torso)
        self._hip_y_min = min(self._hip_y_min, hp[1])
        self._hip_y_max = max(self._hip_y_max, hp[1])

        # 팔꿈치 각도
        arm = _side(lm, (L.LEFT_SHOULDER, L.LEFT_ELBOW, L.LEFT_WRIST),
                    (L.RIGHT_SHOULDER, L.RIGHT_ELBOW, L.RIGHT_WRIST))
        if arm is not None:
            e = joint_angle(p(arm[0]), p(arm[1]), p(arm[2]))
            self._elbow_min = min(self._elbow_min, e)
            self._elbow_max = max(self._elbow_max, e)
            self._elbow_n += 1

        # 손목이 머리(코) 위로 올라갔는지 (이미지 y는 아래로 증가)
        wr = _side(lm, (L.LEFT_WRIST,), (L.RIGHT_WRIST,))
        if wr is not None and _vis(lm, L.NOSE):
            self._wrist_n += 1
            if lm[wr[0]][1] < lm[L.NOSE][1]:
                self._wrist_oh_n += 1

        # 무릎 각도 (양쪽) + 비대칭
        kl = _vis(lm, L.LEFT_HIP, L.LEFT_KNEE, L.LEFT_ANKLE)
        kr = _vis(lm, L.RIGHT_HIP, L.RIGHT_KNEE, L.RIGHT_ANKLE)
        knees = []
        if kl:
            knees.append(joint_angle(p(L.LEFT_HIP), p(L.LEFT_KNEE), p(L.LEFT_ANKLE)))
        if kr:
            knees.append(joint_angle(p(L.RIGHT_HIP), p(L.RIGHT_KNEE), p(L.RIGHT_ANKLE)))
        if knees:
            working = min(knees)  # 더 굽은(작업) 무릎
            self._knee_min = min(self._knee_min, working)
            self._knee_max = max(self._knee_max, working)
            self._knee_n += 1
            if len(knees) == 2:
                self._knee_asym_sum += abs(knees[0] - knees[1])
                self._knee_asym_n += 1
        # 발목 앞뒤 스태거: 좌우 발목의 수직(y) 간격 (런지의 지속적 특징)
        if _vis(lm, L.LEFT_ANKLE, L.RIGHT_ANKLE):
            self._ankle_stag_sum += abs(lm[L.LEFT_ANKLE][1] - lm[L.RIGHT_ANKLE][1])
            self._ankle_stag_n += 1

    def predict(self) -> Tuple[Optional[str], float, Dict[str, float]]:
        """(운동명|None, 신뢰도 0~1, 점수표) 반환."""
        if self._n < MIN_FRAMES:
            return None, 0.0, {}

        torso_mean = self._torso_sum / self._n
        elbow_rom = (self._elbow_max - self._elbow_min) if self._elbow_n else 0.0
        knee_rom = (self._knee_max - self._knee_min) if self._knee_n else 0.0
        hinge_rom = (self._torso_max - self._torso_min)
        hip_rom = (self._hip_y_max - self._hip_y_min)
        wrist_oh = (self._wrist_oh_n / self._wrist_n) if self._wrist_n else 0.0
        knee_asym = (self._knee_asym_sum / self._knee_asym_n) if self._knee_asym_n else 0.0
        ankle_stag = (self._ankle_stag_sum / self._ankle_stag_n) if self._ankle_stag_n else 0.0

        s = {"squat": 0.0, "pushup": 0.0, "plank": 0.0,
             "deadlift": 0.0, "lunge": 0.0, "overhead_press": 0.0}

        # 1) 몸통 방향
        if torso_mean < 40:        # 누운/수평 자세
            s["pushup"] += 1.0
            s["plank"] += 1.0
        else:                      # 선 자세
            for k in ("squat", "deadlift", "lunge", "overhead_press"):
                s[k] += 0.6

        # 2) 팔꿈치 가동범위
        if elbow_rom > 40:
            s["pushup"] += 1.0
            s["overhead_press"] += 0.8
        else:
            s["plank"] += 0.6

        # 3) 손목 머리 위
        if wrist_oh > 0.3:
            s["overhead_press"] += 1.5

        # 4) 무릎 가동범위
        if knee_rom > 40:
            s["squat"] += 1.0
            s["lunge"] += 0.8
            s["deadlift"] += 0.4
            # 좌우 무릎이 대칭이면(평균 비대칭 작음) 스쿼트 가능성↑
            if knee_asym < 15:
                s["squat"] += 0.6
        else:
            s["plank"] += 0.3
            s["overhead_press"] += 0.3

        # 5) 런지: 지속적 무릎 비대칭(평균) 또는 발 앞뒤 스태거
        if knee_asym > 22 or ankle_stag > 0.12:
            s["lunge"] += 1.2

        # 6) 몸통 힌지(각도 변화 큼) + 선 자세 → 데드리프트
        if hinge_rom > 25 and torso_mean > 45:
            s["deadlift"] += 1.0

        # 7) 엉덩이 수직 이동 크고 상체 곧음 → 스쿼트
        if hip_rom > 0.12 and torso_mean > 55:
            s["squat"] += 0.6

        if sum(s.values()) <= 0:
            return None, 0.0, s
        ordered = sorted(s.values(), reverse=True)
        top = ordered[0]
        second = ordered[1] if len(ordered) > 1 else 0.0
        best = max(s, key=s.get)
        # 신뢰도 = 1등이 2등보다 얼마나 우세한가 (1.0 = 단독, 0.5 = 동률)
        confidence = top / (top + second) if (top + second) > 0 else 0.0
        return best, confidence, s
