# fitness-form-coach

웹캠으로 운동 자세를 실시간 분석하고 평가·코칭하는 에이전트.

MediaPipe Pose 로 관절 좌표를 뽑아 **로컬에서 관절 각도·렙 카운팅·자세 이슈를 실시간
처리**하고, **렙 완료/유지 종료 시점에만** 구조화된 요약을 Claude 에이전트에 보내
한국어 코칭과 자세 점수를 받습니다. 매 프레임 LLM 을 호출하지 않으므로 화면이
끊기지 않고 비용도 낮습니다.

## 동작 구조

```
웹캠 프레임
   │
   ▼
MediaPipe Pose ──► 관절 좌표(33점)
   │
   ▼
관절 각도 계산 + 렙 카운팅 상태머신 (로컬, 매 프레임)
   │  (렙 완료 / 유지 종료 시에만)
   ▼
구조화된 요약(JSON) ──► Claude 코치 (백그라운드 스레드, 비동기)
   │                         · 프롬프트 캐싱(고정 시스템 프롬프트)
   │                         · 구조화 출력(form_score / cues / ...)
   ▼
화면 오버레이(스켈레톤 + 점수 + 코칭 큐)
```

- **로컬 처리**: `src/pose`, `src/exercises` — 각도/렙/룰 기반 이슈 감지
- **LLM 코칭**: `src/coach` — `claude-opus-4-8` 기본, 프롬프트 캐싱 + structured output
- **UI**: `src/ui/overlay.py` — 한글 텍스트(PIL) 렌더링

## 지원 운동

스쿼트 · 푸시업 · 플랭크 · 데드리프트 · 런지 · 오버헤드 프레스 (기본 제공).
Claude 시스템 프롬프트에 헬스장 운동 전반의 체크포인트 레퍼런스가 들어 있어,
새 운동을 추가할 때는 `src/exercises` 에 클래스를 하나 만들고 `registry.py` 에
등록하면 됩니다.

## 설치

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
cp .env.example .env          # 그리고 ANTHROPIC_API_KEY 입력
```

> `ANTHROPIC_API_KEY` 가 없어도 실행됩니다. 이 경우 로컬 규칙 기반 코치로 폴백합니다
> (점수·교정 큐 제공, 자연어 품질만 낮음).

## 실행

```bash
python main.py --exercise squat
python main.py -e pushup --camera 1
```

실행 중 키:

| 키 | 동작 |
|----|------|
| `1` `2` `3` `4` `5` `6` | 스쿼트 / 푸시업 / 플랭크 / 데드리프트 / 런지 / 오버헤드프레스 전환 |
| `q` 또는 `ESC` | 종료 |

## 모델 선택

기본 모델은 가장 똑똑한 `claude-opus-4-8` 입니다. 렙마다 호출하는 실시간 루프라
더 빠른 응답/낮은 비용을 원하면 `.env` 의 `COACH_MODEL` 을 바꾸세요:

- `claude-sonnet-4-6` — 빠르고 저렴, 균형
- `claude-haiku-4-5` — 가장 빠르고 저렴

## 새 운동 추가하기

```python
# src/exercises/lunge.py
from .base import Exercise, FormIssue, visible
from ..pose import landmarks as L
from ..pose.angles import joint_angle

class Lunge(Exercise):
    name = "lunge"
    name_ko = "런지"
    primary_metric = "front_knee_angle"
    down_threshold = 110.0
    up_threshold = 160.0

    def compute_metrics(self, lm):
        ...   # 각도 계산
    def check_form(self, metrics):
        ...   # FormIssue 리스트 반환
```

그 뒤 `registry.py` 의 `_REGISTRY` 에 추가하면 끝입니다.

## 한계 / 참고

- 단일 웹캠 2.5D 추정이라 깊이(z) 정확도에 한계가 있습니다. 측면에서 촬영하면
  스쿼트·데드리프트의 힌지/깊이 판단이 가장 정확합니다.
- 임계값(각도 기준)은 일반 성인 기준의 출발점입니다. 사용자에 맞게 각 운동 클래스의
  `down_threshold`/`up_threshold` 와 `check_form` 기준을 조정하세요.
