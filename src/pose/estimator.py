"""MediaPipe Pose 래퍼 (Tasks API). 프레임 → 랜드마크 딕셔너리.

최신 mediapipe(0.10.x)는 레거시 `mp.solutions.pose` 를 제거하고 Tasks API
(`PoseLandmarker`)로 전환했다. 이 모듈은 Tasks API 를 사용하며, 모델 번들
(pose_landmarker_lite.task)이 없으면 최초 실행 시 자동으로 내려받는다.

랜드마크 33점 인덱스는 BlazePose 그대로라 src/pose/landmarks.py 상수가 유효하다.
"""
import os
import urllib.request
from typing import Optional

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import config

from .types import Landmarks

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models",
)
_MODEL_PATH = os.path.join(_MODEL_DIR, "pose_landmarker_lite.task")

# 스켈레톤 시각화용 연결(상체/팔/다리 위주, 33점 인덱스 기준)
_POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),     # 어깨/팔
    (11, 23), (12, 24), (23, 24),                          # 몸통
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),      # 왼다리
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),      # 오른다리
]


def _ensure_model() -> str:
    if not os.path.exists(_MODEL_PATH):
        os.makedirs(_MODEL_DIR, exist_ok=True)
        print(f"[pose] 모델 다운로드 중… → {_MODEL_PATH}")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("[pose] 모델 다운로드 완료.")
    return _MODEL_PATH


class PoseEstimator:
    def __init__(self) -> None:
        model_path = _ensure_model()
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=config.POSE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.POSE_MIN_TRACKING_CONFIDENCE,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        # VIDEO 모드는 단조 증가하는 타임스탬프가 필요. 실제 시간과 무관한
        # 내부 카운터로 항상 유효한 값을 보장한다(추적 안정화 용도).
        self._ts_ms = 0
        self._last_norm = None   # 마지막 정규화 랜드마크 리스트 (그리기용)

    def process(self, frame_bgr) -> Optional[Landmarks]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._ts_ms += 33
        result = self.landmarker.detect_for_video(mp_image, self._ts_ms)

        if not result.pose_landmarks:
            self._last_norm = None
            return None

        person = result.pose_landmarks[0]
        self._last_norm = person
        lm: Landmarks = {}
        for i, p in enumerate(person):
            vis = getattr(p, "visibility", 1.0)
            lm[i] = (p.x, p.y, p.z, vis)
        return lm

    def draw_skeleton(self, frame_bgr) -> None:
        """마지막 process() 결과의 스켈레톤을 frame 위에 직접 그린다."""
        if not self._last_norm:
            return
        h, w = frame_bgr.shape[:2]
        pts = [(int(p.x * w), int(p.y * h)) for p in self._last_norm]
        for a, b in _POSE_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(frame_bgr, pts[a], pts[b], (0, 200, 255), 2)
        for x, y in pts:
            cv2.circle(frame_bgr, (x, y), 3, (0, 120, 255), -1)

    def close(self) -> None:
        self.landmarker.close()
