"""Claude 기반 코칭 에이전트 + 비동기 디스패처 + 룰 기반 폴백.

- 렙/세트 요약(dict)을 받아 Coaching(구조화 결과)을 반환한다.
- 시스템 프롬프트는 프롬프트 캐싱(ephemeral)으로 매 호출 재과금을 피한다.
- 실시간 루프가 멈추지 않도록 AsyncCoach 가 별도 스레드에서 호출한다.
- ANTHROPIC_API_KEY 가 없으면 RuleBasedCoach 로 자동 폴백한다.
"""
from __future__ import annotations

import json
import queue
import threading
from typing import Callable, Dict, Optional

import config

from .schema import SYSTEM_PROMPT, Coaching


class RuleBasedCoach:
    """API 키가 없을 때 쓰는 간단한 로컬 코치. detected_issues 기반."""

    def coach(self, summary: Dict) -> Coaching:
        is_set = summary.get("event") == "set"
        # 세트 요약은 recurring_issues, 그 외(렙/유지)는 detected_issues 사용
        issues = (summary.get("recurring_issues") if is_set
                  else summary.get("detected_issues")) or []
        majors = [i for i in issues if i.get("severity") == "major"]
        minors = [i for i in issues if i.get("severity") == "minor"]

        if majors:
            severity, score = "major", 45
        elif minors:
            severity, score = "minor", 70
        else:
            severity, score = "good", 92

        cues = [i["message_ko"] for i in (majors + minors)][:3]
        if not cues:
            cues = ["좋은 자세입니다. 이대로 유지하세요."]

        if is_set:
            total = summary.get("total_reps", summary.get("total_holds", 0))
            headline = f"{total}개 완료 · " + (
                "교정 포인트 있어요" if majors else
                "좋은 세트였어요" if severity == "good" else "조금만 다듬어요")
            encouragement = "다음 세트도 화이팅!"
        else:
            headline = "교정이 필요해요" if majors else (
                "거의 완벽해요" if severity == "good" else "조금만 다듬어요")
            encouragement = "잘하고 있어요, 계속 가봅시다!"

        return Coaching(
            form_score=score,
            severity=severity,
            headline=headline,
            cues=cues,
            encouragement=encouragement,
        )


class ClaudeCoach:
    """Claude API 코치. 프롬프트 캐싱 + 구조화 출력."""

    def __init__(self, model: str, api_key: str) -> None:
        import anthropic  # 지연 임포트

        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key)
        # 캐싱되는 안정적 시스템 프롬프트 (마지막 블록에 cache_control)
        self.system = [{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }]

    def coach(self, summary: Dict) -> Coaching:
        user_text = (
            "다음 운동 데이터를 분석해 코칭해 주세요.\n"
            + json.dumps(summary, ensure_ascii=False, indent=2)
        )
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=512,
            system=self.system,
            thinking={"type": "disabled"},  # 실시간 지연 최소화
            messages=[{"role": "user", "content": user_text}],
            output_format=Coaching,
        )
        return response.parsed_output


def build_coach():
    """환경에 맞는 코치 인스턴스를 만든다."""
    if config.ANTHROPIC_API_KEY:
        try:
            return ClaudeCoach(config.COACH_MODEL, config.ANTHROPIC_API_KEY)
        except Exception as e:  # 임포트/초기화 실패 시 폴백
            print(f"[coach] Claude 초기화 실패 → 룰 기반으로 폴백: {e}")
    else:
        print("[coach] ANTHROPIC_API_KEY 없음 → 룰 기반 코치 사용")
    return RuleBasedCoach()


class AsyncCoach:
    """코칭 요청을 백그라운드 스레드에서 처리. 메인 루프는 최신 결과만 읽는다."""

    def __init__(self, coach, on_result: Optional[Callable[[Coaching, Dict], None]] = None):
        self._coach = coach
        self._on_result = on_result
        self._q: "queue.Queue[Optional[Dict]]" = queue.Queue(maxsize=8)
        self._latest: Optional[Coaching] = None
        self._lock = threading.Lock()
        self._busy = False
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def submit(self, summary: Dict) -> None:
        """요약을 코칭 큐에 넣는다. 큐가 꽉 차면 가장 오래된 것을 버린다."""
        try:
            self._q.put_nowait(summary)
        except queue.Full:
            try:
                self._q.get_nowait()
                self._q.put_nowait(summary)
            except queue.Empty:
                pass

    def _worker(self) -> None:
        while True:
            summary = self._q.get()
            if summary is None:
                break
            with self._lock:
                self._busy = True
            try:
                result = self._coach.coach(summary)
                with self._lock:
                    self._latest = result
                if self._on_result:
                    self._on_result(result, summary)
            except Exception as e:
                print(f"[coach] 코칭 호출 실패: {e}")
            finally:
                with self._lock:
                    self._busy = False

    @property
    def latest(self) -> Optional[Coaching]:
        with self._lock:
            return self._latest

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    def stop(self) -> None:
        self._q.put(None)
