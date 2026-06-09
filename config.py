"""전역 설정. 환경 변수(.env)에서 읽어온다."""
import os

from dotenv import load_dotenv

load_dotenv()

# Claude
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# 실시간 렙 단위 코칭이라 기본은 opus-4-8(최고 성능). 더 낮은 지연/비용이 필요하면
# COACH_MODEL 을 claude-sonnet-4-6 / claude-haiku-4-5 로 바꾼다.
COACH_MODEL = os.getenv("COACH_MODEL", "claude-opus-4-8")

# 웹캠
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))

# MediaPipe Pose
POSE_MODEL_COMPLEXITY = int(os.getenv("POSE_MODEL_COMPLEXITY", "1"))  # 0/1/2
POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_TRACKING_CONFIDENCE = 0.5

# 한글 렌더링용 폰트 (Windows 기본 맑은 고딕). 없으면 자동 탐색.
KOREAN_FONT_PATH = os.getenv("KOREAN_FONT_PATH", r"C:\Windows\Fonts\malgun.ttf")
