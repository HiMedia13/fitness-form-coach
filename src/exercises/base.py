"""운동 베이스 클래스 + 렙 카운팅 상태머신.

설계
----
- 각 운동은 `compute_metrics()` 에서 관절 각도 등 측정값(dict)을 만든다.
- 렙(rep) 기반 운동은 '주 측정값(primary metric)' 하나의 위/아래 임계로
  down → up 전이를 감지해 렙을 센다. (예: 스쿼트 무릎 각도)
- 각 렙의 최저/최고점, 템포, 폼 이슈를 모아 RepResult 로 반환한다.
- 플랭크 같은 등척성(hold) 운동은 IsometricExercise 를 상속해 시간 기반으로 평가.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..pose.types import Landmarks


@dataclass
class FormIssue:
    code: str          # 머신 식별용 코드 (예: "knee_valgus")
    message_ko: str    # 화면/요약용 짧은 한글 설명
    severity: str      # "minor" | "major"


@dataclass
class RepResult:
    rep_index: int
    metrics: Dict[str, float]       # 렙 종료 시점 대표 측정값
    depth: float                    # 주 측정값의 최저점(가동범위 깊이 지표)
    tempo_s: float                  # 렙 소요 시간(초)
    issues: List[FormIssue] = field(default_factory=list)


@dataclass
class HoldResult:
    duration_s: float
    issues: List[FormIssue] = field(default_factory=list)


VISIBILITY_THRESHOLD = 0.5


def visible(lm: Landmarks, *idxs: int) -> bool:
    """주어진 랜드마크들이 충분히 잘 보이는지."""
    return all(lm.get(i, (0, 0, 0, 0))[3] >= VISIBILITY_THRESHOLD for i in idxs)


class Exercise:
    """렙 기반 운동의 베이스."""

    name: str = "exercise"
    name_ko: str = "운동"
    primary_metric: str = ""        # 렙 카운팅에 쓰는 측정값 키
    down_threshold: float = 0.0     # 이 값 미만이면 '아래' 단계
    up_threshold: float = 0.0       # 이 값 초과면 '위' 단계
    mode: str = "rep"               # "rep" | "hold"

    def __init__(self) -> None:
        self.rep_count = 0
        self._phase = "up"          # "up" | "down"
        self._rep_start_t: Optional[float] = None
        self._rep_min = float("inf")
        self._last_metrics: Dict[str, float] = {}

    # --- 하위 클래스가 구현 ---
    def compute_metrics(self, lm: Landmarks) -> Optional[Dict[str, float]]:
        raise NotImplementedError

    def check_form(self, metrics: Dict[str, float]) -> List[FormIssue]:
        return []

    # --- 공통 렙 상태머신 ---
    def update(self, lm: Landmarks, t: float) -> Optional[RepResult]:
        metrics = self.compute_metrics(lm)
        if metrics is None:
            return None
        self._last_metrics = metrics
        value = metrics.get(self.primary_metric)
        if value is None:
            return None

        result: Optional[RepResult] = None

        if self._phase == "up" and value < self.down_threshold:
            # 내려가기 시작
            self._phase = "down"
            self._rep_start_t = t
            self._rep_min = value
        elif self._phase == "down":
            self._rep_min = min(self._rep_min, value)
            if value > self.up_threshold:
                # 다 올라옴 → 렙 완료
                self._phase = "up"
                self.rep_count += 1
                tempo = (t - self._rep_start_t) if self._rep_start_t else 0.0
                issues = self.check_form({**metrics, "_depth": self._rep_min})
                result = RepResult(
                    rep_index=self.rep_count,
                    metrics=metrics,
                    depth=self._rep_min,
                    tempo_s=round(tempo, 2),
                    issues=issues,
                )
                self._rep_min = float("inf")
        return result

    def reset_counts(self) -> None:
        """세트 종료 후 다음 세트를 위해 카운터/상태를 초기화한다."""
        self.rep_count = 0
        self._phase = "up"
        self._rep_start_t = None
        self._rep_min = float("inf")

    @property
    def last_metrics(self) -> Dict[str, float]:
        return self._last_metrics

    @property
    def phase(self) -> str:
        return self._phase


class IsometricExercise(Exercise):
    """플랭크 등 자세 유지 운동. 렙 대신 유지 시간을 측정."""

    mode = "hold"

    def __init__(self) -> None:
        super().__init__()
        self._hold_start_t: Optional[float] = None
        self.hold_duration = 0.0

    def reset_counts(self) -> None:
        super().reset_counts()
        self._hold_start_t = None
        self.hold_duration = 0.0

    def is_in_position(self, metrics: Dict[str, float]) -> bool:
        raise NotImplementedError

    def update(self, lm: Landmarks, t: float) -> Optional[HoldResult]:
        metrics = self.compute_metrics(lm)
        if metrics is None:
            # 자세 인식 실패 → 유지 종료
            return self._finish(t)
        self._last_metrics = metrics

        if self.is_in_position(metrics):
            if self._hold_start_t is None:
                self._hold_start_t = t
            self.hold_duration = t - self._hold_start_t
            return None
        return self._finish(t)

    def _finish(self, t: float) -> Optional[HoldResult]:
        if self._hold_start_t is None:
            return None
        duration = t - self._hold_start_t
        self._hold_start_t = None
        self.hold_duration = 0.0
        if duration < 1.0:
            return None
        return HoldResult(
            duration_s=round(duration, 1),
            issues=self.check_form(self._last_metrics),
        )
