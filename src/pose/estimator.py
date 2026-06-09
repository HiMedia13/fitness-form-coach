"""MediaPipe Pose 래퍼. 프레임 → 랜드마크 딕셔너리."""
from typing import Optional

import cv2
import mediapipe as mp

import config

from .types import Landmarks


class PoseEstimator:
    def __init__(self) -> None:
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.pose = self.mp_pose.Pose(
            model_complexity=config.POSE_MODEL_COMPLEXITY,
            min_detection_confidence=config.POSE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.POSE_MIN_TRACKING_CONFIDENCE,
        )
        self._raw = None  # 마지막 mediapipe 결과 (스켈레톤 그리기용)

    def process(self, frame_bgr) -> Optional[Landmarks]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.pose.process(rgb)
        self._raw = result

        if not result.pose_landmarks:
            return None

        lm: Landmarks = {}
        for i, p in enumerate(result.pose_landmarks.landmark):
            lm[i] = (p.x, p.y, p.z, p.visibility)
        return lm

    def draw_skeleton(self, frame_bgr) -> None:
        """마지막 process() 결과의 스켈레톤을 frame 위에 그린다."""
        if self._raw and self._raw.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame_bgr,
                self._raw.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_styles.get_default_pose_landmarks_style(),
            )

    def close(self) -> None:
        self.pose.close()
