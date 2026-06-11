# fitness-form-coach

웹캠(실시간) 또는 녹화된 영상으로 운동 자세를 분석하고 평가·코칭하는 에이전트.

MediaPipe Pose 로 관절 좌표를 뽑아 **로컬에서 관절 각도·렙 카운팅·자세 이슈를 실시간
처리**하고, **렙 완료/유지 종료 시점에만** 구조화된 요약을 Claude 에이전트에 보내
한국어 코칭과 자세 점수를 받습니다. 매 프레임 LLM 을 호출하지 않으므로 화면이
끊기지 않고 비용도 낮습니다.

## 동작 구조

```
웹캠/영상 프레임
   │
   ▼
MediaPipe Pose ──► 관절 좌표(33점)
   │
   ▼
관절 각도 계산 + 렙 카운팅 상태머신 (로컬, 매 프레임)
   │  (렙 완료 / 유지 종료 / 세트 종료 시에만)
   ▼
구조화된 요약(JSON) ──► Claude 코치 (백그라운드 스레드, 비동기)
   │                         · 프롬프트 캐싱(고정 시스템 프롬프트)
   │                         · 구조화 출력(form_score / cues / ...)
   ▼
화면 오버레이(스켈레톤 + 점수 + 코칭 큐)
```

- **로컬 처리**: `src/pose`, `src/exercises` — 각도/렙/룰 기반 이슈 감지
  (MediaPipe **Tasks API**(`PoseLandmarker`) 사용. 최신 mediapipe는 레거시
  `solutions` API를 제거했다. 모델 번들은 최초 실행 시 `models/` 에 자동 다운로드)
- **LLM 코칭**: `src/coach` — `claude-opus-4-8` 기본, 프롬프트 캐싱 + structured output.
  렙 단위는 단일 호출, **세트 단위는 도구 사용 에이전트 루프**(렙 궤적을 직접 파고듦)
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
# 실시간 웹캠 (세트 종료 시 운동 자동 인식, 기본 켜짐)
python main.py --exercise squat
python main.py -e pushup --camera 1

# 녹화된 영상 파일 분석
python main.py -e squat --video clips/set1.mp4

# 자동 인식 끄기
python main.py -e squat --no-auto
```

영상 분석 시에는 처리 속도와 무관하게 템포가 정확하도록 **영상 자체 타임라인
(프레임/FPS)** 을 기준으로 시간을 계산합니다. (웹캠은 벽시계 기준, 거울 모드 적용)

실행 중 키:

| 키 | 동작 |
|----|------|
| `1` `2` `3` `4` `5` `6` | 스쿼트 / 푸시업 / 플랭크 / 데드리프트 / 런지 / 오버헤드프레스 전환 |
| `s` | 세트 종료 → 세트 종합 코칭 요청 |
| `a` | 운동 자동 인식 ON/OFF 토글 |
| `space` | 일시정지/재생 (영상 분석 시) |
| `q` 또는 `ESC` | 종료 |

## 운동 자동 인식

세트 동안 매 프레임 포즈 특징(몸통 방향, 팔꿈치·무릎 가동범위, 손목 머리 위 여부,
좌우 무릎 비대칭, 몸통 힌지 범위 등)을 누적하고, **세트가 끝나면 어떤 운동이었는지
자동으로 추정**합니다(`src/recognition.py`, ML 없이 휴리스틱).

- 확신이 충분하면(2등 대비 우세) **다음 세트용 운동으로 자동 전환**합니다.
  → 사용자가 매번 `1`~`6` 으로 고르지 않아도 됩니다.
- 인식 결과는 세트 요약(`detected_exercise`, `detection_confidence`)에 담겨
  Claude 코칭에도 전달됩니다. 카운팅에 쓴 운동과 다르면 코치가 그 점을 짚어줍니다.
- 직접 `1`~`6` 으로 고른 전환은 사용자 의도로 보고 자동 전환하지 않습니다.
- `--no-auto` 로 시작하거나 실행 중 `a` 로 끌 수 있습니다 (화면 우상단 `AUTO`/`수동` 표시).

> 단일 카메라 휴리스틱이라 측면 촬영에서 가장 잘 동작합니다. 첫 세트는 `--exercise`
> 로 지정한 운동으로 카운팅되고, 세트 종료 후부터 자동 보정됩니다.

## 코칭 단위

| 단위 | 시점 | 방식 | 내용 |
|------|------|------|------|
| **렙** | 렙 완료 즉시 | Claude 단일 호출 (빠름) | 그 렙의 점수·교정 큐 (실시간 피드백) |
| **유지** | 플랭크 등 유지 종료 | Claude 단일 호출 | 유지 시간·자세 평가 |
| **세트** | `s` 키 또는 **휴식 자동 감지**(약 12초) | **에이전트(도구 사용 루프)** | 렙별 궤적을 직접 파고들어 원인 분석 후 코칭 |

세트가 바뀌면(운동 전환·휴식·`s`) 렙 카운터가 0으로 초기화되고 화면 상단에 세트 번호가
표시됩니다.

### 세트 심층 분석 — 에이전트(도구 사용 루프)

렙·유지 코칭은 이미 계산된 숫자 요약을 Claude가 자연어로 바꾸는 **단일 호출**입니다.
반면 **세트 종료 시에는 진짜 에이전트 루프**가 돕니다 — Claude가 요약만 보고 끝내지
않고, 직접 도구를 호출해 각 렙의 관절 각도 궤적을 파고든 뒤 결론을 냅니다
(`src/coach/agent.py`의 `AgentCoach`, `src/coach/tools.py`).

```
세트 종료 → AgentCoach
   │  개요(JSON) 제시 + 도구 제공
   ▼
Claude ──► list_reps()              # 전체 렙 훑어보기
       ──► get_rep_stats(rep)       # 의심 렙의 관절별 min/max/avg
       ──► get_rep_trajectory(rep, metric)  # 특정 각도의 시간순 궤적
       ──► (필요한 만큼 반복하며 원인 파악)
       ──► submit_coaching(...)     # 근거를 모은 뒤 최종 코칭 제출
```

도구는 렙의 **프레임별 궤적**(하강·바닥·상승 모양, 좌우 비대칭, 후반 렙 깊이 저하 등)을
노출하므로, Claude가 "3번째 렙부터 무릎 바닥값이 커진다 = 피로로 깊이가 얕아짐" 같은
패턴을 스스로 찾아 코칭합니다. (도구는 **계산된 각도 데이터**를 읽습니다. 원본 영상까지
직접 보게 하려면 비전 기반으로 확장 가능)

> 비용·지연 때문에 에이전트 루프는 세트 종료에만 씁니다. 렙 단위 즉시 피드백은 빠른
> 단일 호출을 유지합니다. API 키가 없으면 세트도 룰 기반 폴백으로 동작합니다.

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

## 헤드리스 테스트

디스플레이 없이(서버/CI) 영상 파이프라인을 검증하려면:

```bash
python tools/test_headless.py <영상경로> [운동명]
```

포즈 인식률, 렙 카운팅, 운동 자동 인식, 세트 요약, 코칭을 콘솔로 출력합니다.
실제 스쿼트 시연 영상으로 검증한 결과 213프레임 전부 포즈 인식, 렙 2회 정확 카운트,
자동 인식 `squat`(신뢰도 0.67)으로 동작을 확인했습니다.

## 한계 / 참고

- 단일 웹캠 2.5D 추정이라 깊이(z) 정확도에 한계가 있습니다. 측면에서 촬영하면
  스쿼트·데드리프트의 힌지/깊이 판단이 가장 정확합니다.
- 임계값(각도 기준)은 일반 성인 기준의 출발점입니다. 사용자에 맞게 각 운동 클래스의
  `down_threshold`/`up_threshold` 와 `check_form` 기준을 조정하세요.
