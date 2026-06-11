"""세트 단위 추적 + 세트 요약 집계.

여러 렙(또는 유지)을 한 세트로 모아, 세트 종료 시 전체 경향을 집계한 요약 dict 를
만든다. 이 요약은 event="set" 으로 Claude 코치에 전달돼 세트 종합 코칭을 받는다.

세트 종료는 (1) 수동(키), (2) 일정 시간 무동작(휴식) 자동 감지 로 트리거된다.
무동작 판정에 쓰는 시간 t 는 호출 측이 넘기며, 웹캠은 벽시계·영상은 영상 타임라인
초 단위를 쓰므로 두 경우 모두 일관되게 동작한다.
"""
from collections import Counter
from typing import Dict, List, Optional

from .exercises.base import HoldResult, RepResult


def build_set_summary(
    exercise_name: str,
    set_index: int,
    reps: List[RepResult],
    holds: List[HoldResult],
) -> Dict:
    summary: Dict = {
        "exercise": exercise_name,
        "event": "set",
        "set_index": set_index,
    }

    if reps:
        depths = [r.depth for r in reps]
        tempos = [r.tempo_s for r in reps]
        # 반복된 자세 이슈 집계 (빈도 높은 순)
        counter: Counter = Counter()
        meta: Dict[str, tuple] = {}
        for r in reps:
            for issue in r.issues:
                counter[issue.code] += 1
                meta[issue.code] = (issue.message_ko, issue.severity)
        summary.update({
            "total_reps": len(reps),
            # depth = 주 관절 최저 각도(작을수록 깊게 수행). 최소/최대로 일관성 파악.
            "avg_depth": round(sum(depths) / len(depths), 1),
            "min_depth": round(min(depths), 1),
            "max_depth": round(max(depths), 1),
            "avg_tempo_s": round(sum(tempos) / len(tempos), 2),
            "per_rep_depth": [round(d, 1) for d in depths],
            "recurring_issues": [
                {
                    "code": code,
                    "message_ko": meta[code][0],
                    "severity": meta[code][1],
                    "count": n,
                }
                for code, n in counter.most_common()
            ],
        })

    if holds:
        durations = [h.duration_s for h in holds]
        summary.update({
            "total_holds": len(holds),
            "total_hold_s": round(sum(durations), 1),
            "longest_hold_s": round(max(durations), 1),
        })

    return summary


class SetTracker:
    """현재 세트의 렙/유지를 누적하고, 세트 종료 시 요약을 만든다."""

    def __init__(self, idle_timeout_s: float = 12.0) -> None:
        self.idle_timeout_s = idle_timeout_s
        self.set_index = 1
        self._reps: List[RepResult] = []
        self._holds: List[HoldResult] = []
        self._last_active_t: Optional[float] = None

    @property
    def has_data(self) -> bool:
        return bool(self._reps or self._holds)

    @property
    def reps(self) -> List[RepResult]:
        """현재 세트의 렙 기록 스냅샷 (에이전트 코치 도구용)."""
        return list(self._reps)

    def add_rep(self, result: RepResult, t: float) -> None:
        self._reps.append(result)
        self._last_active_t = t

    def add_hold(self, result: HoldResult, t: float) -> None:
        self._holds.append(result)
        self._last_active_t = t

    def should_auto_finish(self, t: float) -> bool:
        """마지막 동작 이후 idle_timeout_s 초 이상 무동작이면 세트 자동 종료."""
        return (
            self.has_data
            and self._last_active_t is not None
            and (t - self._last_active_t) > self.idle_timeout_s
        )

    def finish(self, exercise_name: str) -> Optional[Dict]:
        """세트 요약을 만들고 다음 세트를 위해 초기화. 데이터 없으면 None."""
        if not self.has_data:
            return None
        summary = build_set_summary(
            exercise_name, self.set_index, self._reps, self._holds)
        self.set_index += 1
        self._reps = []
        self._holds = []
        self._last_active_t = None
        return summary
